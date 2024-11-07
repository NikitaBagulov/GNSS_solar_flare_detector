import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from astropy.time import Time
from pathlib import Path
import datetime
from utils import get_latlon, retrieve_data, _UTC
from enum import Enum
from sunpy.net import attrs as a
import sunpy.map
import sunpy.data.sample
from astropy.coordinates import SkyCoord
import astropy.units as u
import pickle
import numpy as np
from IndexCalculator import IndexCalculator


class FlarePosition(Enum):
    CENTRAL = "central"
    MEDIUM = "medium"
    EDGE = "edge"
    INVISIBLE = "invisible"

    def classify_flare(hpc_x, hpc_y):
        if -300 <= hpc_x <= 300 and -300 <= hpc_y <= 300:
            return FlarePosition.CENTRAL
        elif -600 <= hpc_x <= 600 and -600 <= hpc_y <= 600:
            return FlarePosition.MEDIUM
        elif -960 <= hpc_x <= 960 and -960 <= hpc_y <= 960:
            return FlarePosition.EDGE
        else:
            return FlarePosition.INVISIBLE
        
def plot_terminator(ax, time=None, color="black", alpha=0.5):
        """
        Plot a fill on the dark side of the planet (without refraction).

        Parameters
        ----------
            ax: axes of matplotlib.plt
                of matplotlib.plt to plot on
            time : datetime
                The time to calculate terminator for. Defaults to datetime.utcnow()
        """
        lat, lon = get_latlon(time)
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

def plot_single_map(filename, output_file, time_key_str, index_ratios_file, flare_data_file, goes_file, tr_file):
    
    

    with open(index_ratios_file, 'rb') as f:
        index_ratios = pickle.load(f)

    with open(flare_data_file, 'rb') as f:
        flare_list = pickle.load(f)

    with open(goes_file, 'rb') as f:
        goes = pickle.load(f)

    with open(tr_file, 'rb') as f:
        tr = pickle.load(f)


    time_key = datetime.datetime.fromisoformat(time_key_str)
    data_points = retrieve_data(filename)[time_key.replace(tzinfo=_UTC)]

    vmin, vmax = 0.0, 0.5
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["blue", "cyan", "yellow", "red"])
    flare_travel_time = datetime.timedelta(minutes=8.5)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[4, 1, 1], width_ratios=[2, 1],hspace=0.3, wspace=0.4)
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_map.set_extent([-180, 180, -90, 90])
    ax_map.coastlines()
    ax_map.set_title(f'Data map at {time_key.strftime("%Y-%m-%d %H:%M:%S")}', fontsize=16)
    
    latitudes = [point[0] for point in data_points]
    longitudes = [point[1] for point in data_points]
    values = [point[2] for point in data_points]
    
    scatter = ax_map.scatter(longitudes, latitudes, c=values, cmap=cmap, s=10, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(scatter, ax=ax_map, fraction=0.046 * (ax_map.get_position().height / ax_map.get_position().width), pad=0.04)
    cbar.set_label('Value')

    plot_terminator(ax_map, time_key)

            # График индексов
    times, ratios = zip(*index_ratios)
    ax_index = fig.add_subplot(gs[1, 0])
    ax_index.plot(times, ratios, label='Индекс', color='orange', linewidth=2)
    ax_index.set_xlabel('Time', fontsize=14)
    ax_index.set_ylabel('Index', fontsize=14)
    ax_index.axvline(x=time_key, color='red', linestyle='-', label='Current time')
    ax_index.annotate(
                        f'      Current time', xy=(time_key, max(ratios)),
                        xytext=(time_key, max(ratios) * 1.1),
                        arrowprops=dict(facecolor="red", shrink=0.05, width=1, headwidth=6),
                        fontsize=10, ha='center', rotation='vertical'
                    )
    ax_index.set_xlim(tr.start.to_datetime(), tr.end.to_datetime())

            # График солнечной активности (GOES XRS)
    ax_goes = fig.add_subplot(gs[2, 0])
    goes.plot(axes=ax_goes)
    ax_goes.set_title('Solar Activity (GOES XRS)')
    ax_goes.axvline(x=time_key, color='red', linestyle='-', label='Current time')
    ax_goes.annotate(text="      Current time",xy=(time_key, max(ratios)),
                        xytext=(time_key, max(ratios) * 1.1),
                        arrowprops=dict(facecolor="red", shrink=0.05, width=1, headwidth=6),
                        fontsize=10, ha='center', rotation='vertical'
                    )
    ax_goes.set_xlim(tr.start.to_datetime(), tr.end.to_datetime())
    ax_goes.set_yscale('log')
    flare_levels = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    for level in flare_levels:
        ax_goes.axhline(y=level, color='gray', linestyle='--', linewidth=0.5)
    ax_goes.set_yticks(flare_levels)
    ax_goes.get_yaxis().set_major_formatter(plt.LogFormatter(base=10))
            
        
            # Обработка вспышек
    for flare in flare_list:
        flare_time = flare['start_time']
        flare_peak_time = flare.get('peak_time')
        flare_end_time = flare.get('end_time')
        flare_class = flare['class']
        flare_position = FlarePosition.classify_flare(flare['hpc_x'], flare['hpc_y'])

                # Цвет для позиции вспышки
        flare_color = {
                    FlarePosition.CENTRAL: 'blue',
                    FlarePosition.MEDIUM: 'orange',
                    FlarePosition.EDGE: 'yellow',
                }.get(flare_position, 'gray')

                # Время начала вспышки
        if flare_time:
            ax_index.axvline(x=flare_time, color=flare_color, linestyle='--')
                
                # Пиковое время
        if flare_peak_time:
            ax_index.axvline(x=flare_peak_time, color=flare_color, linestyle='--')
            ax_index.annotate(
                        f'      {flare_class} (Sun)', xy=(flare_peak_time, max(ratios)),
                        xytext=(flare_peak_time, max(ratios) * 1.1),
                        arrowprops=dict(facecolor=flare_color, shrink=0.05, width=1, headwidth=6),
                        fontsize=10, ha='center', rotation='vertical'
                    )
                    # Время прибытия на Землю
            flare_arrival_time = flare_peak_time + flare_travel_time
            ax_index.axvline(x=flare_arrival_time, color=flare_color, linestyle='-')
            ax_index.annotate(
                        f'      {flare_class} (Earth)', xy=(flare_arrival_time, max(ratios)),
                        xytext=(flare_arrival_time, max(ratios) * 1.1),
                        arrowprops=dict(facecolor=flare_color, shrink=0.05, width=1, headwidth=6),
                        fontsize=10, ha='center', rotation='vertical'
                    )

                # Время окончания вспышки
        if flare_end_time:
            ax_index.axvline(x=flare_end_time, color=flare_color, linestyle='--')
            ax_index.axvspan(flare_time, flare_end_time, color=flare_color, alpha=0.3)
    ax_map.grid()
    ax_index.grid()
    ax_goes.grid()

    

)
    sun_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)
    wcs = sun_map.wcs
    ax_sun = fig.add_subplot(gs[0, 1], projection=wcs)
    sun_map.plot(axes=ax_sun)
    sun_map.draw_limb()
    for flare in flare_list:
        flare_lat = flare['hpc_y']
        flare_lon = flare['hpc_x']
        coord = SkyCoord(flare_lon*u.arcsec, flare_lat*u.arcsec, frame=sun_map.coordinate_frame)
        ax_sun.plot_coord(coord, 'o-')
    
    

    ax_sun.set_title('Sun with Flare Positions')
    # ax_sun.set_theta_zero_location('S')  # Опционально: выставить направление "юг" для наглядности
    ax_sun.grid()
    # Сохраняем результат

    roti_values_day = [point[2] for point in data_points if IndexCalculator.is_daytime(point[0], point[1], time_key) and not np.isnan(point[2])]
    roti_values_night = [point[2] for point in data_points if not IndexCalculator.is_daytime(point[0], point[1], time_key) and not np.isnan(point[2])]

    # Убедимся, что у нас есть данные для построения гистограммы
    if roti_values_day or roti_values_night:
        # Задаем интервалы для гистограммы
        roti_bins = np.histogram_bin_edges(roti_values_day + roti_values_night, bins=20)  # Объединяем оба списка, чтобы интервалы были одинаковыми

        # Вычисляем частоту значений ROTI для дневных и ночных точек
        roti_counts_day, _ = np.histogram(roti_values_day, bins=roti_bins)
        roti_counts_night, _ = np.histogram(roti_values_night, bins=roti_bins)

        # Создание графика
        ax_roti = fig.add_subplot(gs[1, 1])
        width = np.diff(roti_bins)  # Ширина каждого интервала

        # Отображаем гистограммы для дневных и ночных значений
        
        ax_roti.bar(roti_bins[:-1], roti_counts_night, width=width, color='blue', align='edge', label='Nighttime', alpha=0.7)
        ax_roti.bar(roti_bins[:-1], roti_counts_day, width=width, color='orange', align='edge', label='Daytime')

        # Настройка осей и заголовков
        ax_roti.set_xlabel('Index ROTI', fontsize=14)
        ax_roti.set_ylabel('Frequency', fontsize=14)
        ax_roti.set_title('Distribution of Index ROTI (Daytime vs Nighttime)', fontsize=16)
        ax_roti.legend()
    else:
        print("Нет данных для построения графика.")

    subsolat_lat, subsolat_lon = get_latlon(time_key)  # Получение подсолнечной точки на текущий момент
    distances = [
        IndexCalculator.great_circle_distance_rad(subsolat_lat, subsolat_lon, lat, lon) 
        for lat, lon, _ in data_points
    ]

    # Определение дневных и ночных точек
    day_points = [dist for dist, point in zip(distances, data_points) if IndexCalculator.is_daytime(point[0], point[1], time_key)]
    night_points = [dist for dist, point in zip(distances, data_points) if not IndexCalculator.is_daytime(point[0], point[1], time_key)]
    ax_distance = fig.add_subplot(gs[2, 1])
    ax_distance.hist(day_points, bins=20, alpha=0.7, label='Day', color='gold')
    ax_distance.hist(night_points, bins=20, alpha=0.7, label='Night', color='navy')
    ax_distance.set_xlabel('Distance from Subsolar Point', fontsize=14)
    ax_distance.set_ylabel('Frequency', fontsize=14)
    ax_distance.set_title('Distance Distribution: Day vs Night', fontsize=16)
    ax_distance.legend()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    data_file = sys.argv[1]
    output_file = sys.argv[2]
    time_key_str = sys.argv[3]
    index_ratios_file = sys.argv[4]
    flare_data_file = sys.argv[5]
    goes_file = sys.argv[6]
    tr_file = sys.argv[7]
    
    plot_single_map(data_file, output_file, time_key_str, index_ratios_file, flare_data_file, goes_file, tr_file)
