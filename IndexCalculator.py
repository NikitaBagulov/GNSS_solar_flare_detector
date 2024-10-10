import h5py
import numpy as np
from numpy.typing import NDArray
import datetime
import math
from dateutil import tz
from numpy import sin, cos, arccos, pi, arcsin
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sunpy.net import Fido
from sunpy.net import attrs as a
from astropy.time import Time
from pathlib import Path
import matplotlib.lines as mlines

def moving_average(data, window_size=3):
    """Функция для вычисления скользящего среднего."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

RE_meters = 6371000

TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
_UTC = tz.gettz('UTC')
DAYS = 365.25
TILT = 23.44
MAG1 = 9
LON_DEGREES_HOUR = 15.

class IndexCalculator:
    def __init__(self, file: str, start_date: datetime.datetime, duration_minutes: int = 60):
        self.file = file
        self.start_date = start_date.replace(tzinfo=_UTC)
        self.duration_minutes = duration_minutes
        self.times = [start_date + datetime.timedelta(minutes=i) for i in range(duration_minutes)]
        self.data = self.retrieve_data()
        self.index_ratios = self.calculate_ratios() 

    def get_flare_data(self):
        """
        Получает данные о солнечных вспышках для заданного периода времени.
        Использует библиотеку SunPy для получения данных о вспышках через Fido.
        Возвращает список всех вспышек за день, которые входят в диапазон self.times.
        """
        # Извлекаем год, месяц и день из стартовой даты
        year, month, day = self.start_date.year, self.start_date.month, self.start_date.day

        # Создаем tstart и tend
        tstart = f"{year}/{month:02d}/{day:02d}"
        tend = f"{year}/{month:02d}/{day+1:02d}"  # Один день после для полного захвата

        # Параметры поиска солнечных вспышек
        event_type = "FL"
        result = Fido.search(a.Time(tstart, tend), a.hek.EventType(event_type))

        # Извлекаем результаты
        hek_results = result['hek']
        filtered_results = hek_results["event_starttime", "event_peaktime", "event_endtime", "fl_goescls"]

        # Определяем функцию для вычисления величины вспышки, учитывая все варианты
        def get_flare_magnitude(flare):
            fl_goescls = flare.get('fl_goescls', '')
            if len(fl_goescls) > 1:
                return ord(fl_goescls[0]) + float(fl_goescls[1:])
            elif fl_goescls:  # Если есть хотя бы один символ (например, класс без числа)
                return ord(fl_goescls[0])
            else:
                return -float('inf')  # Присваиваем минимальное значение, если класс пустой

        # Преобразуем self.times в список дат
        time_range_start = min(self.times)
        time_range_end = max(self.times)

        # Сортируем вспышки по величине
        by_magnitude = sorted(filtered_results, key=get_flare_magnitude, reverse=True)

        # Используем множество для отслеживания уникальных вспышек по времени начала
        unique_flares = set()

        # Формируем список всех вспышек, которые попадают в диапазон self.times и имеют название
        flare_list = []
        for flare in by_magnitude:
            flare_class = flare.get('fl_goescls', '').strip()
            flare_time = Time(flare['event_starttime']).to_datetime()
            flare_peak = Time(flare['event_peaktime']).to_datetime()
            flare_end = Time(flare['event_endtime']).to_datetime()

            # Проверяем, есть ли название класса вспышки и входит ли она в диапазон self.times
            if flare_class and (time_range_start <= flare_time <= time_range_end) and (time_range_start <= flare_end <= time_range_end):
                if flare_time not in unique_flares:  # Проверяем уникальность по времени начала
                    unique_flares.add(flare_time)
                    flare_list.append({
                        'class': flare_class,
                        'start_time': flare_time,
                        'peak_time': flare_peak,
                        'end_time': flare_end
                    })

        # Возвращаем список всех подходящих вспышек
        return flare_list

    @staticmethod
    def get_latlon(time):
        delta = time - datetime.datetime(time.year, 1, 1, 0, 0, 0)
        doy = delta.days
        ut_hour = time.hour + time.minute / 60. + time.second / (60. * 24.)
        lat = - TILT * math.cos(2 * math.pi * ((doy + MAG1)) / DAYS)
        lon = (12.0 - ut_hour) * LON_DEGREES_HOUR
        return lat, lon
    
    @staticmethod
    def great_circle_distance(late, lone, latp, lonp, R=RE_meters):
        """ 
        Calculates arc length
        late, latp: double
            latitudes of two point in sphere in radians
        lone, lonp: double
            longitudes of two point in sphere in radians
        R: double
            radius in meters
        """ 
        if lone < 0:
            lone += 2*pi
        if lonp < 0:
            lonp += 2*pi

        dlon = lonp - lone
        if dlon > 0.:
            if dlon > pi:
                dlon = 2 * pi - dlon
            else:
                pass
        else:
            if dlon < -pi:
                dlon += 2 * pi
            else:
                dlon = -dlon

        cosgamma = sin(late) * sin(latp) + cos(late) * cos(latp) * cos(dlon)
        return R * arccos(cosgamma)
    
    @staticmethod
    def great_circle_distance_rad(late, lone, latp, lonp, R=RE_meters):
        """
        Вычисляет дистанцию по дуге большого круга между двумя точками на сфере.
        
        late, latp: float
            Широты двух точек в градусах.
        lone, lonp: float
            Долготы двух точек в градусах.
        R: float
            Радиус сферы (по умолчанию радиус Земли).
            
        Возвращает:
        -----------
            float
                Дистанция в метрах между двумя точками.
        """ 
        # Преобразование в радианы
        late, lone, latp, lonp = map(math.radians, [late, lone, latp, lonp])

        # Разница долгот
        dlon = lonp - lone
        dlon = (dlon + math.pi) % (2 * math.pi) - math.pi  # Нормализация

        # Вычисление угла между точками с помощью формулы большого круга
        cosgamma = math.sin(late) * math.sin(latp) + math.cos(late) * math.cos(latp) * math.cos(dlon)

        # Защита от ошибок округления
        cosgamma = min(1, max(-1, cosgamma))

        return R * math.acos(cosgamma)
    def is_daytime(self, lat, lon, time):
        late, lone = self.get_latlon(time)
        # print(late, lone)
        dist = self.great_circle_distance_rad(late, lone, lat, lon)
        # print(dist)
        return dist < (math.pi / 2 * RE_meters)

    def retrieve_data(self) -> dict[datetime.datetime, NDArray]:
        f_in = h5py.File(self.file, 'r')
        data = {}
        times = list(f_in['data'])[:]
        
        with tqdm(total=len(times), desc="Загрузка данных из файла") as progress:
            for str_time in times:
                time = datetime.datetime.strptime(str_time, TIME_FORMAT).replace(tzinfo=tz.gettz('UTC'))
                data[time] = f_in['data'][str_time][:]
                progress.update(1)

        return data

    def calculate_index(self, points):
        """
        Вычисляет индекс I для списка точек с помощью векторизации.
        :param points: список точек, где каждая точка содержит (расстояние, значение)
        :return: сумма индексов для списка точек
        """
        if len(points) == 0:
            return 0.0

        R = np.array([p[0] for p in points])
        values = np.array([p[1] for p in points])
        # eps=1e-10
        # I = (1 / (1 + np.log(R + eps))) * values
        I = (1 / (1 + R)) * values
        I = np.round(I, 10)
        I = np.nan_to_num(I, nan=0.0)
        return np.sum(I)
        

    def calculate_ratios(self):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.process_time, self.times))
        return results

    def process_time(self, time):
        time_key = time.replace(tzinfo=_UTC)
        points = self.data.get(time_key, [])
        total_points = len(points)
        days = []
        nights = []
        count = 0
        with tqdm(total=total_points, desc=f"Обработка точек для {time.strftime('%Y-%m-%d %H:%M:%S')}", leave=False) as point_progress:
            for point in points:
                late, lone = self.get_latlon(time)
                Re = self.great_circle_distance(late, lone, point[0], point[1])

                if self.is_daytime(point[0], point[1], time):
                    days.append((Re, point[2]))
                else:
                    nights.append((Re, point[2]))
                count+=1
                point_progress.update(1)  # Обновляем прогресс для каждой точки
        total_index_day = self.calculate_index(days)
        total_index_night = self.calculate_index(nights)

        if total_index_night != 0 or count != 0:
            day_weight = len(days)/count
            night_weight = len(nights)/count
            weighted_day = total_index_day/day_weight   
            weighted_night = total_index_night/night_weight
            ratio = weighted_day/weighted_night
        else:
            ratio = 0.0

        return (time, ratio)

    def plot_and_save_all_maps(self):
        """Рисует и сохраняет карты данных для всех временных меток с графиками индексов."""
        flare_list = self.get_flare_data()
        vmin, vmax = 0.0, 0.1  # Минимальное и максимальное значения для colorbar
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["blue", "yellow", "red"])
        folder_name = Path(f"{self.start_date.strftime('%Y%m%d')}_full")
        folder_name.mkdir(parents=True, exist_ok=True)
        for time in self.times:
            time_key = time.replace(tzinfo=_UTC)
            if time_key not in self.data:
                print(f"Нет данных для времени: {time}")
                continue

            # Получаем данные: lat, lon, value
            data_points = self.data[time_key]
            
            latitudes = [point[0] for point in data_points]  # Широта
            longitudes = [point[1] for point in data_points]  # Долгота
            values = [point[2] for point in data_points]       # Значение

            times, ratios = [], []
            for t, ratio in self.index_ratios:
                times.append(t)
                ratios.append(ratio)
            
            ratio = ratios[0]  # Берем первый найденный индекс

            # Создание общей фигуры с двумя подграфиками
            fig = plt.figure(figsize=(12, 10))
            # Увеличиваем отступ между подграфиками
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.4)  # Увеличен отступ до 0.6

            # Создаем проекцию карты для верхнего подграфика
            ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
            ax.set_extent([-180, 180, -90, 90])  # Установка диапазона
            ax.coastlines()
            ax.set_title(f'Data map at a point in time {time_key.strftime("%Y-%m-%d %H:%M:%S")}', fontsize=16)
            
            # Построение карты с цветами по значениям на верхнем subplot
            scatter = ax.scatter(longitudes, latitudes, c=values, cmap=cmap, s=10, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            
            # Добавление colorbar
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046 * (ax.get_position().height / ax.get_position().width), pad=0.04)
            cbar.set_label('Value')
            self.plot_terminator(ax, time)

            # Построение графика индексов на нижнем subplot
            ax_index = fig.add_subplot(gs[1])
            ax_index.plot(times, ratios, label='Индекс', color='orange', linewidth=2)
            # ax_index.set_title('График индексов', fontsize=16, pad=70)  # Добавляем отступ между заголовком и графиком
            ax_index.set_xlabel('Time', fontsize=14)
            ax_index.set_ylabel('Index', fontsize=14)
            # ax_index.axvline(x=time_key, color='red', linestyle='--', label='Текущее время')



            colors = list(mcolors.TABLEAU_COLORS.keys())
            legend_handles = [] 
            for i, flare in enumerate(flare_list):
                flare_time = flare['start_time']
                flare_peak_time = flare.get('peak_time')  # Время пика вспышки
                flare_end_time = flare.get('end_time')  # Время конца вспышки
                flare_class = flare['class']

                flare_color = mcolors.TABLEAU_COLORS[colors[i % len(colors)]]

                if flare_time:
                    ax_index.axvline(x=flare_time, color=flare_color, linestyle='--', label=f'Начало вспышки {flare_class}')

                # Отрисовка времени пика вспышки (с аннотацией класса)
                if flare_peak_time:
                    ax_index.axvline(x=flare_peak_time, color=flare_color, linestyle='--', label=f'Пик вспышки {flare_class}')
                    ax_index.annotate(
                        f'{flare_class}',  # Только класс вспышки у пика
                        xy=(flare_peak_time, max(ratios)), 
                        xytext=(flare_peak_time, max(ratios)), 
                        arrowprops=dict(facecolor=flare_color, shrink=0.05), 
                        textcoords='data',
                        fontsize=12,
                        ha='center'
                    )
                    # legend_handles.append(mlines.Line2D([0], [0], color=flare_color, linestyle='--', label=f'{flare_class}'))

                # Отрисовка времени конца вспышки
                if flare_end_time:
                    ax_index.axvline(x=flare_end_time, color=flare_color, linestyle='--', label=f'Конец вспышки {flare_class}')

                # Выделим область вспышки цветом
                if flare_time and flare_end_time:
                    ax_index.axvspan(flare_time, flare_end_time, color=flare_color, alpha=0.3, label=f'Область вспышки {flare_class}')
            ax_index.axvline(x=time_key, color='red', linestyle='-', label='Current time')
            ax_index.annotate(
                'Current time',
                xy=(time_key, min(ratios)), 
                xytext=(time_key, max(ratios)), 
                arrowprops=dict(facecolor='red', shrink=0.05),
                textcoords='data',
                fontsize=12,
                ha='center',
                rotation='vertical')   

            # legend_handles.append(mlines.Line2D([0], [0], color='red', linestyle='--', label='Текущее время'))  # Добавляем в легенду
            # ax_index.legend(handles=legend_handles, fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))  # Переносим легенду вправо
            ax_index.grid()
            # Сохранение карты и графика индексов как PNG
            filename = f"map_with_index_{time_key.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(f"{time_key.strftime('%Y%m%d')}_full/{filename}", bbox_inches='tight')
            plt.close(fig)  # Закрытие фигуры для освобождения памяти
            print(f"Сохранена карта и график индексов для времени {time_key} в файл {filename}")

    def plot_terminator(self, ax, time=None, color="black", alpha=0.5):
        """
        Plot a fill on the dark side of the planet (without refraction).

        Parameters
        ----------
            ax: axes of matplotlib.plt
                of matplotlib.plt to plot on
            time : datetime
                The time to calculate terminator for. Defaults to datetime.utcnow()
        """
        lat, lon = self.get_latlon(time)
        pole_lng = lon
        if lat > 0:
            pole_lat = -90 + lat
            central_rot_lng = 180
        else:
            pole_lat = 90 + lat
            central_rot_lng = 0

        rotated_pole = ccrs.RotatedPole(pole_latitude=pole_lat,
                                        pole_longitude=pole_lng,
                                        central_rotated_longitude=central_rot_lng)

        x = [-90] * 181 + [90] * 181 + [-90]
        y = list(range(-90, 91)) + list(range(90, -91, -1)) + [-90]
        ax.fill(x, y, transform=rotated_pole,
                color=color, alpha=alpha, zorder=3)


# file_path = "roti_2011_249_-90_90_N_-180_180_E_9caa.h5"
# start_date = datetime.datetime(2011, 9, 6, 21, 57, 0)
# calculator = IndexCalculator(file_path, start_date, 42)
# calculator.plot_and_save_all_maps()

# file_path = "roti_2015_125_-90_90_N_-180_180_E_0dfb.h5"
# start_date = datetime.datetime(2015, 5, 5, 21, 50, 0)
# calculator = IndexCalculator(file_path, start_date, 40)
# calculator.plot_and_save_all_maps()

file_path = "roti_2024_214_-90_90_N_-180_180_E_8ed2.h5"
start_date = datetime.datetime(2024, 8, 1, 0, 0, 0)
calculator = IndexCalculator(file_path, start_date, 1440) #1440
calculator.plot_and_save_all_maps()