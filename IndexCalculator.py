import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
import imageio.v3 as iio
from enum import Enum

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

class FlarePosition(Enum):
    CENTRAL = "Центральная"
    MEDIUM = "Средняя"
    EDGE = "Крайняя"
    INVISIBLE = "Невидимая"

    def classify_flare(hpc_x, hpc_y):
        if -300 <= hpc_x <= 300 and -300 <= hpc_y <= 300:
            return FlarePosition.CENTRAL
        elif -600 <= hpc_x <= 600 and -600 <= hpc_y <= 600:
            return FlarePosition.MEDIUM
        elif -960 <= hpc_x <= 960 and -960 <= hpc_y <= 960:
            return FlarePosition.EDGE
        else:
            return FlarePosition.INVISIBLE

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
        filtered_results = hek_results["event_starttime", "event_peaktime", "event_endtime", "fl_goescls", "hpc_x", "hpc_y"]

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

            # Получаем координаты вспышки
            hpc_x = flare.get('hpc_x', None)
            hpc_y = flare.get('hpc_y', None)

            # Проверяем, есть ли название класса вспышки и входит ли она в диапазон self.times
            if flare_class and (time_range_start <= flare_time <= time_range_end) and (time_range_start <= flare_end <= time_range_end):
                if flare_time not in unique_flares:  # Проверяем уникальность по времени начала
                    unique_flares.add(flare_time)
                    
                    # Проверяем, находится ли вспышка в центральной области солнечного диска (условно)
                    if hpc_x is not None and hpc_y is not None:
                            flare_list.append({
                                'class': flare_class,
                                'start_time': flare_time,
                                'peak_time': flare_peak,
                                'end_time': flare_end,
                                'hpc_x': hpc_x,
                                'hpc_y': hpc_y,
                            })
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

    def calculate_index(self, points, is_day=True):
        """
        Вычисляет индекс I для списка точек с помощью векторизации.
        :param points: список точек, где каждая точка содержит (расстояние, значение)
        :return: сумма индексов для списка точек
        """
        if len(points) == 0:
            return 0.0

        d = np.array([p[0] for p in points])
        values = np.array([p[1] for p in points])
        if is_day:
            I = (1-1/(2*math.pi*RE_meters/4)*d) * values
        else:
            I=values
        I = np.round(I, 10)
        I = np.nan_to_num(I, nan=0.0)
        return np.sum(I)
        

    def calculate_ratios(self):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.process_time, self.times))
        return results

    def great_circle_distance_numpy(self, late, lone, latp, lonp, R=RE_meters):
        """ 
        Calculates arc length. Uses numpy arrays
        late, latp: double
            latitude in radians
        lone, lonp: double
            longitudes in radians
        R: double
            radius
        """ 
        lone[np.where(lone < 0)] = lone[np.where(lone < 0)] + 2*pi
        lonp[np.where(lonp < 0)] = lonp[np.where(lonp < 0)] + 2*pi
        dlon = lonp - lone
        inds = np.where((dlon > 0) & (dlon > pi)) 
        dlon[inds] = 2 * pi - dlon[inds]
        dlon[np.where((dlon < 0) & (dlon < -pi))] += 2 * pi
        dlon[np.where((dlon < 0) & (dlon < -pi))] = -dlon[np.where((dlon < 0) & (dlon < -pi))]
        cosgamma = np.sin(late) * np.sin(latp) + np.cos(late) * np.cos(latp) * np.cos(dlon)
        return R * arccos(cosgamma)

    def process_time(self, time):
        time_key = time.replace(tzinfo=_UTC)
        points = self.data.get(time_key, [])
        total_points = len(points)
        days = []
        nights = []
        count = 0

        latitudes = np.array([point[0] for point in points])
        longitudes = np.array([point[1] for point in points])

        late, lone = self.get_latlon(time)

        late_array = np.full(latitudes.shape, late)
        lone_array = np.full(latitudes.shape, lone)

        distances = self.great_circle_distance_numpy(late_array, lone_array, latitudes, longitudes)

        with tqdm(total=total_points, desc=f"Обработка точек для {time.strftime('%Y-%m-%d %H:%M:%S')}", leave=False) as point_progress:
            for i, point in enumerate(points):
                d = distances[i]
                if self.is_daytime(point[0], point[1], time):
                    days.append((d, point[2]))
                else:
                    nights.append((d, point[2]))
                
                count += 1
                point_progress.update(1)

        total_index_day = self.calculate_index(days)
        total_index_night = self.calculate_index(nights, is_day=False)

        if total_index_night != 0 or count != 0:
            weighted_day = total_index_day / len(days) if len(days) > 0 else 0
            weighted_night = total_index_night / len(nights) if len(nights) > 0 else 0
            ratio = weighted_day / weighted_night if weighted_night > 0 else 0

        return (time, ratio)

    def plot_and_save_all_maps(self):
        """Рисует и сохраняет карты данных для всех временных меток с графиками индексов."""
        flare_list = self.get_flare_data()
        vmin, vmax = 0.0, 0.5  # Минимальное и максимальное значения для colorbar
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["blue","cyan", "yellow", "red"])
        folder_name = Path(f"{self.start_date.strftime('%Y%m%d')}_full")
        folder_name.mkdir(parents=True, exist_ok=True)
        flare_travel_time = datetime.timedelta(minutes=8.5)
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
            ax_index.set_xlabel('Time', fontsize=14)
            ax_index.set_ylabel('Index', fontsize=14)
            # ax_index.axvline(x=time_key, color='red', linestyle='--', label='Текущее время')



            colors = list(mcolors.TABLEAU_COLORS.keys())
            for i, flare in enumerate(flare_list):
                flare_time = flare['start_time']
                flare_peak_time = flare.get('peak_time')  # Время пика вспышки
                flare_end_time = flare.get('end_time')  # Время конца вспышки
                flare_class = flare['class']
                flare_position = FlarePosition.classify_flare(flare['hpc_x'], flare['hpc_y'])

                # flare_color = mcolors.TABLEAU_COLORS[colors[i % len(colors)]]
                if flare_position == FlarePosition.CENTRAL:
                    flare_color = 'Blue'
                elif flare_position == FlarePosition.MEDIUM:
                    flare_color = 'orange'
                elif flare_position == FlarePosition.EDGE:
                    flare_color = 'yellow'
                else:
                    flare_color = 'gray'

                if flare_time:
                    ax_index.axvline(x=flare_time, color=flare_color, linestyle='--', label=f'Start flare {flare_class}')
                    flare_arrival_time = flare_time + flare_travel_time
                    ax_index.axvline(x=flare_arrival_time, color=flare_color, linestyle='-', label=f'Peak (Earth) {flare_class}')
                    ax_index.annotate(
                        f'   {flare_class} (Earth)',  # Только класс вспышки у пика
                        xy=(flare_arrival_time, max(ratios)), 
                        xytext=(flare_arrival_time, max(ratios)), 
                        arrowprops=dict(facecolor=flare_color, shrink=0.05), 
                        textcoords='data',
                        fontsize=12,
                        ha='center',
                        rotation='vertical'
                    )

                # Отрисовка времени пика вспышки (с аннотацией класса)
                if flare_peak_time:
                    ax_index.axvline(x=flare_peak_time, color=flare_color, linestyle='--', label=f'Peak(Sun){flare_class}')
                    ax_index.annotate(
                        f'   {flare_class} (Sun)',  # Только класс вспышки у пика
                        xy=(flare_peak_time, max(ratios)), 
                        xytext=(flare_peak_time, max(ratios)), 
                        arrowprops=dict(facecolor=flare_color, shrink=0.05), 
                        textcoords='data',
                        fontsize=12,
                        ha='center',
                        rotation='vertical'
                    )
                    

                # Отрисовка времени конца вспышки
                if flare_end_time:
                    ax_index.axvline(x=flare_end_time, color=flare_color, linestyle='--', label=f'End flare {flare_class}')

                # Выделим область вспышки цветом
                if flare_time and flare_end_time:
                    ax_index.axvspan(flare_time, flare_end_time, color=flare_color, alpha=0.3, label=f'Flare area {flare_class}')
            ax_index.axvline(x=time_key, color='red', linestyle='-', label='Current time')
            ax_index.annotate(
                '    Current time',
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

        self.create_video_from_maps()

    def create_video_from_maps(self, output_filename="animation.mp4"):
        """Создает видео-анимацию из сохраненных изображений карт."""
        folder_name = Path(f"{self.start_date.strftime('%Y%m%d')}_full")
        image_files = sorted(folder_name.glob("map_with_index_*.png"))  # Собираем все PNG файлы
        
        if not image_files:
            print("Не найдено изображений для создания видео.")
            return

        # Создаем список для хранения изображений
        images = []
        
        for image_file in image_files:
            # Загружаем каждое изображение
            image = iio.imread(image_file)
            image = self.convert_to_rgb(image)  # Читаем изображение
            images.append(image)  # Добавляем изображение в список

        # Создаем видео
        output_path = folder_name / output_filename
        
        # Записываем видео с помощью imageio
        iio.imwrite(output_path, images, fps=2, codec='libx264')  # Устанавливаем частоту кадров
        
        print(f"Видео сохранено в {output_path}")

    def convert_to_rgb(self, image):
        """
        Преобразует изображение в формат RGB, если оно имеет другую форму.
        """
        if len(image.shape) == 2:  # Если изображение черно-белое
            image = np.stack([image] * 3, axis=-1)  # Дублируем каналы для создания RGB
        elif image.shape[-1] == 4:  # Если изображение имеет альфа-канал (RGBA)
            image = image[..., :3]  # Убираем альфа-канал

        return image

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
calculator = IndexCalculator(file_path, start_date, 360) #1440
calculator.plot_and_save_all_maps()
