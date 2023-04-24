from word_embedding.TextToTfIdf import TextToTfIdf
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

class CosineSimilarityConv(TextToTfIdf):
    def __init__(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, text_column: str):
        TextToTfIdf.__init__(self, text_column)
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.text_column = text_column
        # Обучаем TfidfVectorizer только на train данных, для остальных будет сразу transform
        # Для косинусного сходства между TF-IDF матрицами нужно, чтобы они были получены с использованием одного и того же TfidfVectorizer
        self.fit(self.train_data)
        self.cosine_maps = []

    def create_kernel_map(self, kernel_list: list, stride: int, label: str):
        # Для каждого датафрейма свой, поэтому очищаем
        self.cosine_maps.clear()

        # stride - шаг по которому идем через текст, идем сразу по всем строчкам
        for kernel in tqdm(kernel_list):
            # Косинусовая карта активации для конкретно этого ядра
            activation_map = []
            # Переводим в tfidf ядро
            kernel_data = {f'{self.text_column}': [kernel]}
            kernel_df = pd.DataFrame(kernel_data)
            kernel_tfidf = self.transform(kernel_df)
            kernel = kernel.split()

            # Длиной карты активации, все карты активации одного размера, сколько слов за раз берем из текста из всех строк
            ind_end = len(kernel)

            # нам нужно получить длину самого большого предложения
            if label == 'train':
                ind_max = self.train_data[f'{self.text_column}'].str.split().apply(len).max()
            elif label == 'valid':
                ind_max = self.valid_data[f'{self.text_column}'].str.split().apply(len).max()
            elif label == 'test':
                ind_max = self.test_data[f'{self.text_column}'].str.split().apply(len).max()

            for ind_start in range(0, ind_max, stride):
                # именно этот текст (text_split) мы должны перевести в TfIdf и вычислить косинус
                if label == 'train':
                    text_split =  self.train_data[f'{self.text_column}'].str.split().str[ind_start:ind_end]
                elif label == 'valid':
                    text_split =  self.valid_data[f'{self.text_column}'].str.split().str[ind_start:ind_end]
                elif label == 'test':
                    text_split =  self.test_data[f'{self.text_column}'].str.split().str[ind_start:ind_end]

                # объединяем каждую строчку из массива в строку
                text_split = text_split.apply(lambda x: ', '.join(x))
                text_split = pd.DataFrame(text_split)
                text_tfidf = self.transform(text_split)

                # Теперь считаем Косинусное сходство и добавляем его в карту
                similar_cosine_similarity = cosine_similarity(text_tfidf, kernel_tfidf)
                similar_cosine_similarity = [float(x[0]) for x in similar_cosine_similarity.tolist()]
                activation_map.append(similar_cosine_similarity)

                ind_end += len(kernel)

            # Транспонирруем activation_map чтобы привести из [[1,2,3], [2,4,6]]) в [[1, 2], [2, 4], [3, 6]]
            # Один массив -> один текст
            activation_map = [list(row) for row in np.array(activation_map).T]
            # Теперь добавим в общее хранилище для карт (cosine_map)
            self.cosine_maps.append(activation_map)

    def sum_all_cosine_map(self,):
        # Объединяем все что получили в одну
        sum_cosmap = self.cosine_maps[0]
        for cosmap in self.cosine_maps[1:]:
            sum_cosmap = np.add(sum_cosmap, cosmap)
        return sum_cosmap
