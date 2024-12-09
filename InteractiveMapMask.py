import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import numpy as np
import json
from MapMask import MapMask

class InteractiveMapMask(MapMask):
    def __init__(self, lat_cell_size, lon_cell_size):
        super().__init__(lat_cell_size, lon_cell_size)
        self.selected_cells = set()
        self.fig = None  # Хранить ссылку на фигуру
        self.ax = None   # Хранить ссылку на ось
        self.rects = []  # Хранить ссылки на ячейки сетки

    def on_click(self, event):
        """Обработчик клика для выбора/снятия выделения ячеек."""
        if event.inaxes:
            lon, lat = event.xdata, event.ydata

            # Найти ячейку, куда попал клик
            for idx, (min_lat, max_lat, min_lon, max_lon) in enumerate(self.grid):
                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                    if idx in self.selected_cells:
                        self.selected_cells.remove(idx)  # Снятие выделения
                    else:
                        self.selected_cells.add(idx)  # Выбор ячейки
                    break

            # Обновить отображение
            self.update_visualization()

    def visualize_on_map_cartopy(self, load_from_file=False, file_path="selected_cells.txt", points=None):
        """
        Инициализировать отображение карты. 
        Если load_from_file=True, загружает выбранные ячейки из файла.
        Если False, позволяет выбрать ячейки вручную и сохранить их в файл.
        """
        if load_from_file and file_path:
            print(f"Loading selected cells from {file_path}...")
            self.load_selected_cells(file_path)
            return self.get_selected_cells()  # Возврат загруженных ячеек

        print("Initializing plot for manual selection...")
        self.fig, self.ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})

        # Создаем сетку
        self.rects = []
        for idx, cell in enumerate(self.grid):
            min_lat, max_lat, min_lon, max_lon = cell
            rect = plt.Rectangle(
                (min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                linewidth=1, edgecolor='gray', facecolor='none'
            )
            self.ax.add_patch(rect)
            self.rects.append(rect)

        # Отображение точек, если указаны
        if points is not None:
            latitudes = [point[0] for point in points]
            longitudes = [point[1] for point in points]
            values = [point[2] for point in points]

            vmin, vmax = 0.0, 0.5
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["blue", "cyan", "yellow", "red"])

            self.scatter = self.ax.scatter(
                longitudes, latitudes, c=values, cmap=cmap, s=10, vmin=vmin, vmax=vmax,
                transform=ccrs.PlateCarree(), alpha=0.5, zorder=0
            )
            cbar = plt.colorbar(
                self.scatter, ax=self.ax, fraction=0.046 * (self.ax.get_position().height / self.ax.get_position().width), pad=0.04
            )
            cbar.set_label('Value')

        self.ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        self.ax.coastlines()
        plt.title("Interactive Map with Scatter")
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Первоначальная отрисовка
        self.update_visualization()
        plt.show()  # Оставить окно открытым, но не блокировать выполнение кода

        if not load_from_file and file_path:
            self.save_selected_cells(file_path)
            return self.get_selected_cells()


    def update_visualization(self, updated_idx=None):
        """
        Обновить цвета ячеек на основе выбранных. 
        Если указан `updated_idx`, обновляется только одна ячейка.
        """
        if updated_idx is not None:
            # Обновить только одну ячейку
            rect = self.rects[updated_idx]
            if updated_idx in self.selected_cells:
                rect.set_facecolor('yellow')
                rect.set_alpha(0.6)
            else:
                rect.set_facecolor('none')
                rect.set_alpha(0.6)
            rect.stale = True  # Сообщить matplotlib, что элемент изменен
        else:
            # Обновить все ячейки (используется для начальной отрисовки)
            for idx, rect in enumerate(self.rects):
                if idx in self.selected_cells:
                    rect.set_facecolor('yellow')
                    rect.set_alpha(0.6)
                else:
                    rect.set_facecolor('none')
                    rect.set_alpha(0.6)
                rect.stale = True

        # Обновить только измененные элементы
        self.ax.figure.canvas.draw()

    def get_selected_cells(self):
        """Возвращает выбранные ячейки как список."""
        print(list(self.selected_cells))
        return list(self.selected_cells)

    def save_selected_cells(self, file_path):
        """Сохранение выбранных ячеек в файл."""
        with open(file_path, 'w') as f:
            json.dump(list(self.selected_cells), f)
        print(f"Selected cells saved to {file_path}.")

    def load_selected_cells(self, file_path):
        """Загрузка выбранных ячеек из файла."""
        try:
            with open(file_path, 'r') as f:
                self.selected_cells = set(json.load(f))
                # self.update_visualization()  # Обновить отображение после загрузки
                return self.get_selected_cells()
                # print(f"Selected cells loaded from {file_path}.")
        except FileNotFoundError:
            print(f"File {file_path} not found. Starting with an empty selection.")
