import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import numpy as np

class MapMask:
    def __init__(self, lat_cell_size, lon_cell_size):
        """
        Инициализация класса MapMask.

        :param lat_cell_size: Размер ячейки по широте (например, 5 градусов)
        :param lon_cell_size: Размер ячейки по долготе (например, 5 градусов)
        """
        if lat_cell_size <= 0 or lon_cell_size <= 0:
            raise ValueError("Размер ячейки должен быть положительным числом.")
        
        if lat_cell_size > 180 or lon_cell_size > 360:
            raise ValueError("Размер ячейки не может быть больше 180° по широте и 360° по долготе.")
        
        # Убедимся, что шаги позволяют построить корректную сетку
        if lat_cell_size > 180:
            raise ValueError("Шаг по широте не может быть больше 180 градусов.")
        if lon_cell_size > 360:
            raise ValueError("Шаг по долготе не может быть больше 360 градусов.")
        
        self.lat_cell_size = lat_cell_size
        self.lon_cell_size = lon_cell_size
        self.grid = self._generate_grid()

    def _generate_grid(self):
        """
        Генерация сетки карты для всей поверхности Земли.

        :return: Список ячеек [(min_lat, max_lat, min_lon, max_lon), ...]
        """
        grid = []
        # Широта варьируется от -90 до 90
        lat_steps = np.arange(-90, 90, self.lat_cell_size)
        # Долгота варьируется от -180 до 180
        lon_steps = np.arange(-180, 180, self.lon_cell_size)

        for lat in lat_steps:
            for lon in lon_steps:
                grid.append((
                    lat, min(lat + self.lat_cell_size, 90),
                    lon, min(lon + self.lon_cell_size, 180)
                ))
        return grid


    def filter_points(self, points, selected_cells):
        """
        Фильтрация точек, попадающих в выбранные ячейки.

        :param points: Список точек np.array([(lat, lon, data), ...]
        :param selected_cells: Список индексов выбранных ячеек
        :return: Отфильтрованные точки
        """
        masks = np.array([self.grid[i] for i in selected_cells])
        filtered_points = []

        # Извлекаем границы ячеек
        min_lats = masks[:, 0]
        max_lats = masks[:, 1]
        min_lons = masks[:, 2]
        max_lons = masks[:, 3]
        # Извлекаем широты и долготы из точек
        latitudes = np.array([point[0] for point in points])
        longitudes = np.array([point[1] for point in points])

        # Проверяем каждую ячейку
        for min_lat, max_lat, min_lon, max_lon in zip(min_lats, max_lats, min_lons, max_lons):
            # Используем векторизированные условия для фильтрации
            lat_mask = (latitudes >= min_lat) & (latitudes <= max_lat)
            lon_mask = (longitudes >= min_lon) & (longitudes <= max_lon)
            mask = lat_mask & lon_mask
            
            # Добавляем отфильтрованные точки
            filtered_points.append(points[mask])

        # Объединяем все отфильтрованные точки в один массив
        return np.concatenate(filtered_points)


    def visualize_on_map_cartopy(self, selected_cells=None, points=None, save_path=None):
        print("Start plotting...")
        
        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Отображаем ячейки сетки
        for idx, cell in enumerate(self.grid):
            min_lat, max_lat, min_lon, max_lon = cell
            rect = plt.Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                                linewidth=1, edgecolor='gray', facecolor='none')
            ax.add_patch(rect)

            if selected_cells and idx in selected_cells:
                rect.set_facecolor('blue')
                rect.set_alpha(0.6)

        # Отображение точек
        if points is not None:
            latitudes = np.array([point[0] for point in points])
            longitudes = np.array([point[1] for point in points])
            ax.scatter(longitudes, latitudes, color='red', s=50)

        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.coastlines()
        plt.title("Map Visualization")
        print(save_path)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Map saved to {save_path}")

# map_mask = MapMask(lat_cell_size=10, lon_cell_size=20)

# # Выбираем произвольные ячейки (например, с индексами 5, 15, 25)
# selected_cells = [i for i in range(36, 144)] + [i for i in range(216, 275)] + [i for i in range(279, 290)] + [i for i in range(301, 306)]

# # Пример точек (широта, долгота)
# points = [
#     (15, 30, 121),  # Попадает в одну из выбранных ячеек
#     (-15, 60, 123),  # Вне выбранных ячеек
#     (40, -150, 123)  # Тоже вне
# ]

# # Фильтруем точки
# filtered_points = map_mask.filter_points(points, selected_cells)

# # Визуализируем карту
# folium_map = map_mask.visualize_on_map(selected_cells=selected_cells, points=filtered_points)
# # folium_map.save("complex_mask_map.html")
# # folium_map