import sys
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import json
from pathlib import Path
import datetime
from utils import get_latlon, retrieve_data, _UTC
from enum import Enum
from sunpy.net import attrs as a
import pickle

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
        print("Success lat lon", lat, lon)
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

    # Парсинг времени
    time_key = datetime.datetime.fromisoformat(time_key_str)
    data_points = retrieve_data(filename)[time_key.replace(tzinfo=_UTC)]
    # Построение графиков
    vmin, vmax = 0.0, 0.5
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["blue", "cyan", "yellow", "red"])
    flare_travel_time = datetime.timedelta(minutes=8.5)
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[4, 1, 1], hspace=0.3)
    ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
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
    ax_index = fig.add_subplot(gs[1])
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
    ax_goes = fig.add_subplot(gs[2])
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

    # Сохраняем результат
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
