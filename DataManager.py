import os
import requests
import datetime
from pathlib import Path
from typing import List, Dict, Optional
from IndexCalculator import IndexCalculator
from dateutil import parser as date_parser
from concurrent.futures import ThreadPoolExecutor, as_completed


class DataManager:
    def __init__(self, email: str, download_folder: str = "downloaded_files"):
        self.email = email
        self.download_folder = Path(download_folder)
        self.download_folder.mkdir(parents=True, exist_ok=True)
        self.query_results = self.checking_by_mail()

    def checking_by_mail(self) -> List[Dict]:
        """
        Получает все доступные сведения о запросах, сделанных по email.
        Возвращает список словарей с информацией о каждом запросе.
        """
        rq = requests.post(
            "https://simurg.iszf.irk.ru/api",
            json={"method": "check", "args": {"email": self.email}}
        )
        return rq.json()

    def filter_roti_maps(self) -> List[Dict]:
        """
        Фильтрует результаты запросов, оставляя только уникальные записи, 
        которые относятся к ROTI картам и начинаются не раньше 2010 года.
        """
        seen = set()
        filtered_items = []

        for item in self.query_results:
            # Уникальный ключ для проверки на дублирование
            unique_key = (item.get('type'), item.get('product_type', '').lower(), item.get('begin'))
            
            if unique_key in seen:
                continue
            
            if item.get('type') == 'map' and item.get('product_type', '').lower() == 'roti':
                begin_time_str = item.get('begin')
                if begin_time_str:
                    begin_time = date_parser.parse(begin_time_str)
                    if begin_time.year >= 2010:
                        filtered_items.append(item)
                        seen.add(unique_key)
                    else:
                        print(f"Пропущен файл с датой до 2010 года: {begin_time_str}")

        return filtered_items

    def download_file(self, item: Dict, idx: int, total: int):
        """
        Скачивает файл по сформированной ссылке и сохраняет его в заданную папку.
        """
        data_path = item['paths'].get('data')
        if data_path:
            file_name = os.path.basename(data_path)
            download_url = f"https://simurg.space/ufiles/{data_path}"
            local_file_path = self.download_folder / file_name

            # Скачиваем файл
            if not local_file_path.exists():
                print(f"[{idx}/{total}] Скачивание {file_name}...")
                response = requests.get(download_url)
                if response.status_code == 200:
                    with open(local_file_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Файл {file_name} успешно скачан.")
                else:
                    print(f"Ошибка при скачивании файла {file_name}: {response.status_code}")
            else:
                print(f"Файл {file_name} уже существует. Пропускаем скачивание.")

            # Добавляем локальный путь к элементу для дальнейшей обработки
            item['local_file_path'] = str(local_file_path)

    def download_and_process_by_link(self, link: str, download_folder: str = "downloaded_files"):
        """
        Скачивает файл по ссылке, находит его среди запросов с указанной почты,
        и обрабатывает с помощью IndexCalculator.
        """
        # Создаем объект DataManager
        manager = DataManager(email=self.email, download_folder=download_folder)
        
        # Получаем все запросы по email
        all_items = manager.checking_by_mail()
        
        # Находим запрос по идентификатору
        result_id = link.split('=')[-1]  # Извлекаем ID из ссылки
        
        found_item = None
        print(all_items)
        for item in all_items:
            if item.get('id') == result_id:
                found_item = item
                break
        
        if not found_item:
            print(f"Запрос с ID {result_id} не найден среди запросов с почты {self.email}.")
            return
        
        # Скачиваем файл
        manager.download_file(found_item, idx=1, total=1)
        
        # Обрабатываем файл
        manager.process_single_file(found_item, idx=1, total=1)

    def download_files(self, items: List[Dict]):
        """
        Скачивает файлы по сформированным ссылкам и сохраняет их в заданную папку.
        """
        for item in items:
            data_path = item['paths'].get('data')
            if data_path:
                file_name = os.path.basename(data_path)
                download_url = f"https://simurg.space/ufiles/{data_path}"
                local_file_path = self.download_folder / file_name

                # Скачиваем файл
                if not local_file_path.exists():
                    print(f"Скачивание {file_name}...")
                    response = requests.get(download_url)
                    if response.status_code == 200:
                        with open(local_file_path, 'wb') as f:
                            f.write(response.content)
                        print(f"Файл {file_name} успешно скачан.")
                    else:
                        print(f"Ошибка при скачивании файла {file_name}: {response.status_code}")
                else:
                    print(f"Файл {file_name} уже существует. Пропускаем скачивание.")

                # Добавляем локальный путь к элементу для дальнейшей обработки
                item['local_file_path'] = str(local_file_path)

    def process_single_file(self, item: Dict, idx: int, total: int):
        local_file_path = item.get('local_file_path')
        if not local_file_path:
            print(f"Нет локального пути для файла {item['paths'].get('data')}. Пропускаем.")
            return

        begin_time_str = item.get('begin')
        end_time_str = item.get('end')

        if not begin_time_str or not end_time_str:
            print(f"Нет информации о времени начала или конца для файла {local_file_path}. Пропускаем.")
            return

        # Парсим времена начала и конца
        begin_time = date_parser.parse(begin_time_str)
        end_time = date_parser.parse(end_time_str)

        # Вычисляем продолжительность в минутах
        duration_minutes = int((end_time - begin_time).total_seconds() / 60)

        # Обрабатываем файл с помощью IndexCalculator
        print(f"[{idx}/{total}] Обработка файла {local_file_path} с началом {begin_time} и продолжительностью {duration_minutes} минут.")
        calculator = IndexCalculator(local_file_path, begin_time, duration_minutes)
        calculator.plot_and_save_all_maps()

    def process_files(self, items: List[Dict], selected_index: Optional[int] = None):
        """
        Обрабатывает каждый файл с помощью IndexCalculator. Можно выбрать обработку одного файла по индексу.
        """
        total_files = len(items)
        if selected_index is not None:
            # Обрабатываем только один файл по выбранному индексу
            if 0 <= selected_index < total_files:
                self.process_single_file(items[selected_index], selected_index + 1, total_files)
            else:
                print(f"Неверный индекс файла. Допустимый диапазон: 0-{total_files-1}.")
        else:
            # Обрабатываем все файлы параллельно
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.process_single_file, item, idx + 1, total_files)
                    for idx, item in enumerate(items)
                ]

                for future in as_completed(futures):
                    future.result()  # Для отлавливания исключений, если они возникнут.

    def run(self, selected_index: Optional[int] = None):
        """
        Основной метод для запуска процесса получения, скачивания и обработки файлов.
        Можно указать индекс для обработки только одного файла.
        """
        roti_items = self.filter_roti_maps()
        if not roti_items:
            print("Нет доступных ROTI карт для обработки.")
            return

        if selected_index is not None:
            # Обрабатываем только один файл по индексу
            if 0 <= selected_index < len(roti_items):
                self.download_file(roti_items[selected_index], selected_index + 1, 1)
                self.process_files(roti_items, selected_index)

            else:
                print(f"Неверный индекс файла. Допустимый диапазон: 0-{len(roti_items)-1}.")
        else:
            # Скачиваем и обрабатываем все файлы
            self.download_files(roti_items)
            self.process_files(roti_items)
    def process_all_requests(email: str, download_folder: str = "downloaded_files"):
        """
        Создает объект DataManager, получает все доступные запросы и обрабатывает их последовательно.
        """
        # Создаем объект DataManager
        data_manager = DataManager(email=email, download_folder=download_folder)
        
        # Получаем отфильтрованные элементы (только ROTI карты)
        roti_items = data_manager.filter_roti_maps()
        print(roti_items)
        
        if not roti_items:
            print("Нет доступных ROTI карт для обработки.")
            return
        
        # Обрабатываем каждый элемент поочередно
        for idx, item in enumerate(roti_items):
            try:
                print(f"\n--- Обработка элемента {idx + 1} из {len(roti_items)} ---")
                data_manager.download_file(item, idx + 1, len(roti_items))  # Скачиваем файл
                data_manager.process_files(roti_items, idx)  # Обрабатываем файл
            except Exception as e:
                print(f"Ошибка при обработке элемента {idx + 1}: {e}")

# Пример использования:
# data_manager = DataManager(email='simurg30s@simurg.iszf.irk.ru')
# Если нужно обработать только первый файл, можно передать индекс 0.
# data_manager.run(selected_index=0)

# Если нужно обработать все файлы, можно запустить без параметров.
# data_manager.run()
