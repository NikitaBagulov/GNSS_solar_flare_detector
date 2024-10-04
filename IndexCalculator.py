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

        # Сортируем вспышки по величине
        by_magnitude = sorted(filtered_results, key=get_flare_magnitude, reverse=True)

        # Берем первую вспышку с наибольшим классом
        if by_magnitude:
            flare = by_magnitude[0]
            flare_class = flare.get('fl_goescls', 'Unknown')
            flare_time = flare['event_starttime']
            return flare_class, flare_time
        else:
            return None, None

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

        Re = np.array([p[0] for p in points])
        values = np.array([p[1] for p in points])

        I = (1 / (1 - Re)) * values
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
        if total_index_night != 0:
            ratio = total_index_day / total_index_night
        else:
            ratio = 0.0

        return (time, ratio)

    def plot_and_save_all_maps(self):
        """Рисует и сохраняет карты данных для всех временных меток с графиками индексов."""
        flare_class, flare_time = self.get_flare_data()
        if flare_time:
            flare_time = Time(flare_time).to_datetime()
        vmin, vmax = 0.0, 0.1  # Минимальное и максимальное значения для colorbar
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["blue", "yellow", "red"])
        for time in self.times:
            print(time)
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
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.4)  # Увеличенный отступ между подграфиками

            # Создаем проекцию карты для верхнего подграфика
            ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
            ax.set_extent([-180, 180, -90, 90])  # Установка диапазона
            ax.coastlines()
            ax.set_title(f'Карта данных в момент времени {time_key.strftime("%Y-%m-%d %H:%M:%S")}', fontsize=16)
            
            # Построение карты с цветами по значениям на верхнем subplot
            scatter = ax.scatter(longitudes, latitudes, c=values, cmap=cmap, s=10, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            
            # Добавление colorbar
            cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.05)
            cbar.set_label('Значение')
            self.plot_terminator(ax, time)

            # Построение графика индексов на нижнем subplot
            ax_index = fig.add_subplot(gs[1])
            ax_index.plot(times, ratios, label='Индекс', color='orange', linewidth=2)
            ax_index.set_title('График индексов', fontsize=16)
            ax_index.set_xlabel('Время', fontsize=14)
            ax_index.set_ylabel('Индекс', fontsize=14)
            ax_index.axvline(x=time_key, color='red', linestyle='--', label='Текущее время')

            if flare_time and flare_class:
                ax_index.axvline(x=flare_time, color='blue', linestyle='--', label=f'Вспышка {flare_class}')
                ax_index.annotate(f'Вспышка {flare_class}', xy=(flare_time, max(ratios)), xytext=(flare_time, max(ratios) * 1.1),
                                  arrowprops=dict(facecolor='blue', shrink=0.05))

            ax_index.legend()
            ax_index.grid()

            # Сохранение карты и графика индексов как PNG
            filename = f"map_with_index_{time_key.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, bbox_inches='tight')
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

# # file_path = "dtec_2_10_2017_001_-90_90_N_-180_180_E_3d57.h5"
file_path = "roti_2011_249_-90_90_N_-180_180_E_9caa.h5"
start_date = datetime.datetime(2011, 9, 6, 21, 57, 0)
calculator = IndexCalculator(file_path, start_date, 42)
calculator.plot_and_save_all_maps()


# file_path = "roti_2002_204_-90_90_N_-180_180_E_ff0b.h5"
# start_date = datetime.datetime(2002, 7, 23, 0, 3, 0)
# calculator = IndexCalculator(file_path, start_date, 59)
# calculator.plot_and_save_all_maps()