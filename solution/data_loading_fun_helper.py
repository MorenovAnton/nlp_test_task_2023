import json
import pandas as pd

def read_data(data_path: str) -> json:
    # Открываем файл с данными в формате JSON
    with open(data_path, 'r') as file:
        # Загружаем данные из файла
        data = json.load(file)
        return data

def create_dataset(data: json, label = None) -> pd.DataFrame:
    df = pd.DataFrame()
    # создаем списки значений для каждого столбца
    ids = []
    texts = []
    labels = []
    extracted_texts = []
    answer_starts = []
    answer_ends = []
    for item in data:
        ids.append(item["id"])
        texts.append(item["text"])
        labels.append(item["label"])
        if label == 'train':
            # извлекаем значения из словаря extracted_part
            extracted_part = item.get("extracted_part", {})
            extracted_text = str(extracted_part.get("text")[0])
            answer_start = int(extracted_part.get("answer_start")[0])
            answer_end = int(extracted_part.get("answer_end")[0])
            if answer_start == 0 & answer_end == 0:
                extracted_text = ''
            # добавляем значения в списки
            extracted_texts.append(extracted_text)
            answer_starts.append(answer_start)
            answer_ends.append(answer_end)
    # создаем DataFrame из списков значений
    if label == 'train':
        df = pd.DataFrame({
            "id": ids,
            "text": texts,
            "label": labels,
            "extracted_text": extracted_texts,
            "answer_start": answer_starts,
            "answer_end": answer_ends
            })
    elif label == 'test':
        df = pd.DataFrame({
            "id": ids,
            "text": texts,
            "label": labels
            })
    return df



