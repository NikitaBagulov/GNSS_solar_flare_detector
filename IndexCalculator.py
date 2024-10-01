import h5py
import numpy as np
from numpy.typing import NDArray
import datetime
import math
from dateutil import tz
from numpy import sin, cos, arccos, pi, arcsin
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

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
    
    # @staticmethod
    # def calculate_solar_angle(longitude, time):
    #     solar_time = (time.hour + time.minute / 60) * 15
    #     solar_angle = (solar_time + longitude) % 360
    #     return solar_angle

    # def is_daytime(self, lon, time):
    #     solar_angle = self.calculate_solar_angle(lon, time)

    #     object_angle = lon

    #     return (solar_angle < 180 and object_angle < 180) or (solar_angle >= 180 and object_angle >= 180)

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
            results = executor.map(self.process_time, self.times)

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

    # def calculate_ratios(self):
    #     index_ratios = []
    #     total_times = len(self.times)

    #     for idx, time in enumerate(self.times):
    #         print(f"Обработка {idx + 1}/{total_times}: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    #         time_key = time.replace(tzinfo=_UTC)  
    #         points = self.data.get(time_key, [])
    #         days = []
    #         nights = []
    #         for point in points:
    #             late, lone = self.get_latlon(time)
    #             Re = self.great_circle_distance(late, lone, point[0], point[1])

    #             if self.is_daytime(point[1], time):
    #                 days.append((Re, point[2]))
    #             else:
    #                 nights.append((Re, point[2]))

    #         total_index_day = self.calculate_index(days)
    #         total_index_night = self.calculate_index(nights)
    #         if total_index_night != 0:  # Проверяем, чтобы избежать деления на ноль
    #             ratio = total_index_day / total_index_night
    #         else:
    #             ratio = 0.0


    #         index_ratios.append((time, ratio))

    #     return index_ratios


file_path = "dtec_2_10_2017_001_-90_90_N_-180_180_E_3d57.h5"
# file_path = "roti_2011_249_-90_90_N_-180_180_E_9caa.h5"
start_date = datetime.datetime(2017, 1, 1, 0, 0, 0)
import matplotlib.pyplot as plt
calculator = IndexCalculator(file_path, start_date, 1440)
ratios = calculator.calculate_ratios()

# Инициализация списков для времени и значений
times, values = [], []
for time, ratio in ratios:
    times.append(time)
    values.append(ratio)

# Вычисляем сглаженные значения (например, с окном = 5)
window_size = 5
smoothed_values = moving_average(values, window_size=window_size)

# Обрезаем список times, чтобы его длина соответствовала длине сглаженных данных
# (т.к. после сглаживания длина данных уменьшится)
smoothed_times = times[:len(smoothed_values)]

# Построение графиков
plt.figure(figsize=(12, 6))

# График без сглаживания
plt.subplot(2, 1, 1)
plt.plot(times, values, marker='o', label='Без сглаживания')
plt.title('График значений без сглаживания')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.xticks(rotation=45)
plt.grid(True)

# График со сглаживанием
plt.subplot(2, 1, 2)
plt.plot(smoothed_times, smoothed_values, marker='o', color='orange', label=f'Со сглаживанием (окно = {window_size})')
plt.title('График значений со сглаживанием')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()  # Подгоняем элементы графика
plt.show()