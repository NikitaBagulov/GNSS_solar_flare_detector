import h5py
from pathlib import Path
from numpy.typing import NDArray
import datetime
from dateutil import tz
TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
_UTC = tz.gettz('UTC')

import math
DAYS = 365.25
TILT = 23.44
MAG1 = 9
LON_DEGREES_HOUR = 15.

def get_latlon(time):
        delta = time - datetime.datetime(time.year, 1, 1, 0, 0, 0)
        doy = delta.days
        ut_hour = time.hour + time.minute / 60. + time.second / (60. * 24.)
        lat = - TILT * math.cos(2 * math.pi * ((doy + MAG1)) / DAYS)
        lon = (12.0 - ut_hour) * LON_DEGREES_HOUR
        return lat, lon



from numpy import sin, cos, arccos, pi, arcsin

RE_meters = 6371000

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

from numpy import radians

def calculate_solar_angle(longitude, time):
    # Определяем солнечное время (в градусах)
    # 15 градусов = 1 час, для UTC 0h = 0°
    solar_time = (time.hour + time.minute / 60) * 15
    solar_angle = (solar_time + longitude) % 360
    return solar_angle

def is_daytime(lon, time):
    solar_angle = calculate_solar_angle(lon, time)
    
    # Долгота объекта в градусах
    object_angle = lon
    
    # Определяем, на какой стороне
    return (solar_angle < 180 and object_angle < 180) or (solar_angle >= 180 and object_angle >= 180)


def retrieve_data(
    file: str | Path,
    times: list[datetime.datetime] | None = None
) -> dict[datetime.datetime, NDArray]:    
    """
    Retrieves data from HDF file and put in dictionary
    Keys are datetime.datetime values are structured numpy arrays
    file - path to file
    times - times to preserve in output
    """  
    if times is None:
        times = []
    f_in = h5py.File(file, 'r')
    lats = []
    lons = []
    values = []
    data = {}
    for str_time in list(f_in['data'])[:]:
        time = datetime.datetime.strptime(str_time, TIME_FORMAT)
        time = time.replace(tzinfo=time.tzinfo or _UTC)
        if times and not time in times:
            # print(type(time))
            continue
        
        data[time] = f_in['data'][str_time][:]
    return data

times_map2d = [(datetime.datetime(2017, 1, 1) + datetime.timedelta(hours=i)).replace(tzinfo=(datetime.datetime(2017, 1, 1) + datetime.timedelta(hours=i)).tzinfo or _UTC) for i in range(24)]
times = [datetime.datetime(2017, 1, 1) + datetime.timedelta(hours=i) for i in range(24)]
data = retrieve_data("dtec_2_10_2017_001_-90_90_N_-180_180_E_3d57.h5", times=times_map2d)





# print(get_latlon(times[0]))
# print(data[times_map2d[0]])
earth_center = get_latlon(times[11])
points = data[times_map2d[11]]
new_points = []
days = []
nights = []
# print(points)
for point in points:

    late, lone = earth_center[0], earth_center[1]
    Re = great_circle_distance(late, lone, point[0], point[1])
    if is_daytime(point[1], times[11]):
        days.append((Re, point[2]))
    else:
        nights.append((Re, point[2]))
    new_points.append((Re, point[2]))
import numpy as np
def calculate_index(points):
    """
    Вычисляет индекс I для списка точек.

    :param points: список точек, где каждая точка содержит (расстояние, значение)
    :return: сумма индексов для списка точек
    """
    index = 0.0
    for Re, value in points:
        I = (1 / (1 - Re)) * value  # Расчет индекса I
        index+= np.round(I, 10)
        # print(index)
        index = np.nan_to_num(index, nan=0.0)
    # print(index)
    return index
    # return total_index

day_index_sum = calculate_index(days)
night_index_sum = calculate_index(nights)
# Вычисляем отношение
if night_index_sum != 0:
    ratio = day_index_sum / night_index_sum
else:
    ratio = None  # Или обработка случая, когда ночная сумма равна нулю

print(f"Сумма индексов дневных точек: {day_index_sum}")
print(f"Сумма индексов ночных точек: {night_index_sum}")
print(f"Отношение сумм индексов: {ratio}")

import matplotlib.pyplot as plt
def plot_data(data, date):
    """
    Построение простой карты на основе данных.
    
    :param data: Список точек (lat, lon, val)
    :param date: Дата, используемая для названия файла
    """
    # Извлекаем данные
    lats = [point[0] for point in data]
    lons = [point[1] for point in data]
    vals = [point[2] for point in data]

    # Создание фигуры и оси
    fig, ax = plt.subplots(figsize=(10, 5))

    # Создание рассеянного графика с фиксированными границами colorbar
    scatter = ax.scatter(lons, lats, c=vals, cmap='viridis', s=50, edgecolor='k', vmin=-2, vmax=2)

    # Добавление цветовой панели
    cbar = plt.colorbar(scatter, ax=ax, label='Value')
    cbar.set_ticks([-3, 0, 3])  # Установка фиксированных меток на colorbar

    # Заголовок
    ax.set_title(f'Map for {date.strftime("%Y%m%d_%H%M%S")}')
    
    # Установка меток осей
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Сохранение карты
    plt.savefig(f'map_{date.strftime("%Y%m%d_%H%M%S")}.png', dpi=300)
    plt.close()
index_ratios = []
for time in times:

    # Преобразование времени в формат, используемый в ключах data
    time_key = time.replace(tzinfo=_UTC)  # Убираем tzinfo для соответствия с keys в data
    
    points = data.get(time_key, [])
    plot_data(points, time)
    days = []
    nights = []
    
    for point in points:
        late, lone = get_latlon(time)
        Re = great_circle_distance(np.radians(late), np.radians(lone), np.radians(point[0]), np.radians(point[1]))
        
        if is_daytime(point[1], time):
            days.append((Re, point[2]))
        else:
            nights.append((Re, point[2]))

    # Вычисляем индексы
    total_index_day = calculate_index(days)
    total_index_night = calculate_index(nights)
    
    if total_index_night != 0:  # Проверяем, чтобы избежать деления на ноль
        ratio = total_index_day / total_index_night

    index_ratios.append(ratio)


plt.figure(figsize=(12, 6))
plt.plot(times, index_ratios, marker='o', linestyle='-', color='purple', label='Индексы отношений')
plt.xlabel('Время')
plt.ylabel('Отношение индексов')
plt.title('Отношение индексов дневных и ночных точек')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Линия Y=0 для удобства
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


