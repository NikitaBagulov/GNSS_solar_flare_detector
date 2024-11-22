import numpy as np
import sunpy
import datetime
import math

from numpy import sin, cos, arccos, pi
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sunpy.net import Fido
from sunpy.net import attrs as a
from astropy.time import Time
from pathlib import Path
import imageio.v3 as iio
from sunpy.timeseries import TimeSeries
import time as tm

from MapMask import MapMask

from astropy.coordinates import SkyCoord
import astropy.units as u


from utils import get_latlon, retrieve_data, RE_meters, _UTC, TIME_FORMAT
import json
import subprocess
import pickle

class IndexCalculator:
    def __init__(self, file: str, start_date: datetime.datetime, duration_minutes: int = 60):
        self.file = file
        self.start_date = start_date.replace(tzinfo=_UTC)
        self.duration_minutes = duration_minutes
        self.times = [start_date + datetime.timedelta(minutes=i) for i in range(duration_minutes)]
        self.end_date = self.times[-1]
        self.data = retrieve_data(self.file)
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
    @staticmethod
    def is_daytime(lat, lon, time):
        late, lone = get_latlon(time)
        # print(late, lone)
        dist = IndexCalculator.great_circle_distance_rad(late, lone, lat, lon)
        # print(dist)
        return dist < (math.pi / 2 * RE_meters)

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
            # I = (1-1/(2*math.pi*RE_meters/4)*d) * values
            weights = np.maximum(0, 1 - d / (2 * np.pi * RE_meters / 4))
            I = weights * values
        else:
            I=values * np.mean( np.maximum(0, 1 - d / (2 * np.pi * RE_meters / 4)))
            # I=values * np.mean( (1 - d / (2 * np.pi * RE_meters / 4)))
            # I=values
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

        self.map_mask = MapMask(10, 20)
        selected_cells = [i for i in range(36, 144)] + [i for i in range(216, 275)] + [i for i in range(279, 290)] + [i for i in range(301, 306)]
        filtered_points = self.map_mask.filter_points(np.array(points), selected_cells)
        folder_name = Path(f"{self.start_date.strftime('%Y%m%d')}_full")
        folder_name.mkdir(parents=True, exist_ok=True)
        name = time.strftime('%Y_%m_%d_%H_%M_%S')+".png"
        # self.map_mask.visualize_on_map_cartopy(selected_cells=selected_cells, points=filtered_points, save_path=f'./test/{name}')

        latitudes = np.array([point[0] for point in filtered_points])
        longitudes = np.array([point[1] for point in filtered_points])

        late, lone = get_latlon(time)

        late_array = np.full(latitudes.shape, late)
        lone_array = np.full(latitudes.shape, lone)

        distances = self.great_circle_distance_numpy(late_array, lone_array, latitudes, longitudes)
        total_points = len(filtered_points)
        with tqdm(total=total_points, desc=f"Обработка точек для {time.strftime('%Y-%m-%d %H:%M:%S')}", leave=False) as point_progress:
            for i, point in enumerate(filtered_points):
                d = distances[i]
                if self.is_daytime(point[0], point[1], time):
                    days.append((d, point[2]))
                else:
                    nights.append((d, point[2]))
                
                count += 1
                point_progress.update(1)

        total_index_day = self.calculate_index(days)
        total_index_night = self.calculate_index(nights, is_day=False)
        ratio = 0
        if total_index_night != 0 or count != 0:
            weighted_day = total_index_day / len(days) if len(days) > 0 else 0
            weighted_night = total_index_night / len(nights) if len(nights) > 0 else 0
            ratio = weighted_day / weighted_night if weighted_night > 0 else 0

        return (time, ratio)

    def plot_and_save_all_maps(self):
        tr = a.Time(self.times[0].strftime('%Y-%m-%d %H:%M:%S'), self.times[-1].strftime('%Y-%m-%d %H:%M:%S'))
        results = Fido.search(tr, a.Instrument.xrs & a.goes.SatelliteNumber(15) & a.Resolution("avg1m"))
        files = Fido.fetch(results)
        goes = TimeSeries(files)
        print(goes)
        if isinstance(goes, list):
            goes = goes[0] if len(goes) > 0 else []
        print(goes)
        folder_name = Path(f"{self.start_date.strftime('%Y%m%d')}_full")
        folder_name.mkdir(parents=True, exist_ok=True)

        index_ratios_file = folder_name / "index_ratios.pickle"
        with open(index_ratios_file, 'wb') as f:
            pickle.dump(self.index_ratios, f)

        flare_data_file = folder_name / "flare_data.pickle"
        flare_data = self.get_flare_data()
        with open(flare_data_file, 'wb') as f:
            pickle.dump(flare_data, f)

        goes_file = folder_name / "goes_data.pickle"
        with open(goes_file, 'wb') as f:
            pickle.dump(goes, f)

        tr_file = folder_name / "tr.pickle"
        with open(tr_file, 'wb') as f:
            pickle.dump(tr, f)

        flare_x = [flare['hpc_x'] * u.arcsec for flare in flare_data]
        flare_y = [flare['hpc_y'] * u.arcsec for flare in flare_data]

        min_x = min(flare_x)
        max_x = max(flare_x)
        min_y = min(flare_y)
        max_y = max(flare_y)

        buffer = 500 * u.arcsec
        min_x -= buffer
        max_x += buffer
        min_y -= buffer
        max_y += buffer
        start_time = Time(self.start_date.strftime('%Y-%m-%dT%H:%M:%S'), scale='utc', format='isot')
        end_time = Time(self.end_date.strftime('%Y-%m-%dT%H:%M:%S'), scale='utc', format='isot')
        bottom_left = SkyCoord(min_x, min_y, obstime=start_time, observer="earth", frame="helioprojective")
        top_right = SkyCoord(max_x, max_y, obstime=start_time, observer="earth", frame="helioprojective")

        cutout = a.jsoc.Cutout(bottom_left, top_right=top_right, tracking=True)
        
        query = Fido.search(
            a.Time(start_time, end_time),
            a.Wavelength(171*u.angstrom),
            a.Sample(1*u.min),
            a.jsoc.Series.aia_lev1_euv_12s,   
            a.jsoc.Segment.image,
            a.jsoc.Notify('nikita.bagulov.arshan@gmail.com'),
            # cutout
        )
        # files = Fido.fetch(query)
        files = []
        print(query)
        files.sort()
        print(len(self.times), len(files))
        # maps_sun = {key.strftime('%Y-%m-%dT%H:%M:%S'): value for key, value in zip(self.times, files)}
        maps_sun = {}

        maps_sun_file = folder_name / "maps_sun.pickle"
        with open(maps_sun_file, 'wb') as f:
            pickle.dump(maps_sun, f)

        processes = []
        total_files = len(self.times)
        max_processes = 5 

        with tqdm(total=total_files, desc="Прогресс обработки карт") as pbar:
            for i, time in enumerate(self.times, start=1):
                time_key = time.replace(tzinfo=_UTC)
                if time_key not in self.data:
                    print(f"Нет данных для времени: {time}")
                    continue

                output_file = folder_name / f"map_with_index_and_goes_{time_key.strftime('%Y%m%d_%H%M%S')}.png"

                # Запуск процесса
                while len(processes) >= max_processes:
                    # Проверяем завершение процессов
                    for process, index in processes:
                        if process.poll() is not None:  # Проверка завершения
                            pbar.update(1)  # Обновление прогресс-бара
                            processes.remove((process, index))
                            break  # Прерываем цикл для повторной проверки

                    # Добавляем небольшой таймер, чтобы не перегружать процессор
                    tm.sleep(0.5)

                process = subprocess.Popen([
                    "python", "plot_single_map.py",
                    str(self.file),
                    str(output_file),
                    time.isoformat(),
                    str(index_ratios_file),
                    str(flare_data_file),
                    str(goes_file), 
                    str(tr_file), 
                    str(maps_sun_file)
                ])
                processes.append((process, i))

            # Ожидаем завершения оставшихся процессов
            while processes:
                for process, index in processes:
                    if process.poll() is not None:  # Проверка завершения
                        pbar.update(1)  # Обновление прогресс-бара
                        processes.remove((process, index))
                        break  # Прерываем цикл для повторной проверки

                # Добавляем небольшой таймер, чтобы не перегружать процессор
                tm.sleep(0.5)

        # Все процессы завершены, создаем видео
        self.create_video_from_maps(output_filename="animation.mp4")
        print("Все процессы завершены, видео создано.")

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


# file_path = "roti_2011_249_-90_90_N_-180_180_E_9caa.h5"
# start_date = datetime.datetime(2011, 9, 6, 21, 57, 0)
# calculator = IndexCalculator(file_path, start_date, 42)
# calculator.plot_and_save_all_maps()

# file_path = "roti_2015_125_-90_90_N_-180_180_E_0dfb.h5"
# start_date = datetime.datetime(2015, 5, 5, 21, 50, 0)
# calculator = IndexCalculator(file_path, start_date, 40)
# calculator.plot_and_save_all_maps()

# file_path = "roti_2024_214_-90_90_N_-180_180_E_8ed2.h5"
# start_date = datetime.datetime(2024, 8, 1, 0, 0, 0)
# calculator = IndexCalculator(file_path, start_date, 360) #1440
# calculator.plot_and_save_all_maps()
