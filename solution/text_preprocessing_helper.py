import re
import pandas as pd
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def cleaning_text(data: pd.DataFrame, column_name: str):
    #  удалить escape-символы
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub("(\\t)", ' ', str(x)).lower())
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub("(\\r)", ' ', str(x)).lower())
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub("(\\n)", ' ', str(x)).lower())
    # удалим динное нижнее подчеркивание
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub("(__+)", ' ', str(x)).lower())
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub("(--+)", ' ', str(x)).lower())
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub("(~~+)", ' ', str(x)).lower())
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub("(\+\++)", ' ', str(x)).lower())
    # удалим многоточие
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub("(\.\.+)", ' ', str(x)).lower())
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(x)).lower())
    # избавление от точек в конце слов
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub(r"(\.\s+)", ' ', str(x)).lower())
    # удаляме - в конце слов
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub(r"(\-\s+)", ' ', str(x)).lower())
    # удаляем : в конце слов
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub(r"(\:\s+)", ' ', str(x)).lower())
    # избавимся от символа номера
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub(r"№", ' ', str(x)).lower())
    # удаляем любые одиночные символы между двумя пробелами
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub("(\s+.\s+)", ' ', str(x)).lower())
    # удалияем несколько пробелов подряд
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub("(\s+)", ' ',str(x)).lower())
    # заменяем 44-ФЗ и их аналоги на пустое значение
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub(r'\d+[-][a-zA-Zа-яА-Я]\w+', ' ', str(x)).lower())
    # Убираем числа
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub(r'\d+', ' ', str(x)).lower())
    # убираем точки между пробелов
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub(r'\s*\.\s*', ' ', str(x)).lower())
    #  Убираем только те вхождения, которые находятся отдельно от других букв или цифр,
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub(r'\bст\b', '', str(x)).lower())
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub(r'\bбль\b', '', str(x)).lower())
    # убираем символы цетирования
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub(r'[«»“”„”]', '', str(x)).lower())
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub(r'[\\/]', '', str(x)).lower())
    # убираем английский текст
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub(r'[a-zA-Z]+', '', str(x)).lower())
    data.loc[:, column_name] = data[column_name].apply(lambda x: re.sub(r'\s+', ' ', str(x)).lower())


def check_spelling(data: pd.DataFrame, label: str):
    # проверяем правильность написания в тексте и находим правильное написание слова
    morph = pymorphy2.MorphAnalyzer()

    for ind, row in data.iterrows():
        corrected_words_text = []
        corrected_words_extracted_text = []

        text = row.text
        splits_text = text.split()

        if label == 'train':
            extracted_text = row.extracted_text
            splits_extracted_text = extracted_text.split()

            for word in splits_extracted_text:
                parsed_word = morph.parse(word)[0].normal_form
                if parsed_word == word:
                    corrected_words_extracted_text.append(word)
                else:
                    corrected_words_extracted_text.append(parsed_word)
            corrected_words_extracted_text = ' '.join(corrected_words_extracted_text)
            data.loc[ind, 'extracted_text'] = corrected_words_extracted_text

        for word in splits_text:
            parsed_word = morph.parse(word)[0].normal_form
            if parsed_word == word:
                corrected_words_text.append(word)
            else:
                corrected_words_text.append(parsed_word)
        corrected_words_text = ' '.join(corrected_words_text)
        data.loc[ind, 'text'] = corrected_words_text

def remove_stopwords(data: pd.DataFrame, label: str):
    # Удаляем стоп слова
    tokenizer = ToktokTokenizer()
    stopword_list = stopwords.words('russian')
    for ind, row in data.iterrows():
        text = row.text
        tokens_text = tokenizer.tokenize(text)
        tokens_text = [token.strip() for token in tokens_text]
        filtered_text = [token for token in tokens_text if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_text)
        data.loc[ind, 'text'] = filtered_text
        if label == 'train':
            extracted_text = row.extracted_text
            tokens_extracted_text = tokenizer.tokenize(extracted_text)
            tokens_extracted_text = [token.strip() for token in tokens_extracted_text]
            filtered_extracted_text = [token for token in tokens_extracted_text if token.lower() not in stopword_list]
            filtered_extracted_text = ' '.join(filtered_extracted_text)
            data.loc[ind, 'extracted_text'] = filtered_extracted_text

def lem_tokens(data: pd.DataFrame, label: str):
    # Лемматизация
    lemmer = WordNetLemmatizer()
    tokenizer = ToktokTokenizer()

    for ind, row in data.iterrows():
        text_tokens = tokenizer.tokenize(row.text)
        lem_text = [lemmer.lemmatize(token) for token in text_tokens]
        lem_text = ' '.join(lem_text)
        data.loc[ind, 'text'] = lem_text

        if label == 'train':
            extracted_tokens = tokenizer.tokenize(row.extracted_text)
            lem_extracted = [lemmer.lemmatize(token) for token in extracted_tokens]
            lem_extracted = ' '.join(lem_extracted)
            data.loc[ind, 'extracted_text'] = lem_extracted

def find_new_answer_indexes(data: pd.DataFrame):
    # Формируем новые таргет переменные для слов, только для train данных
    tokenizer = ToktokTokenizer()

    for ind, row in data.iterrows():

        text_tokens = tokenizer.tokenize(row.text)

        if row.answer_start == 0 and row.answer_end == 0:
            data.loc[ind, 'new_answer_start'] = 0
            data.loc[ind, 'new_answer_end'] = 0
            continue

        patern_sentence = tokenizer.tokenize(row.extracted_text)
        len_patern_sentence = len(patern_sentence)
        patern_sentence = ' '.join(patern_sentence)
        # Чтобы не выйти за границы при прибавлении
        for i in range(len(text_tokens) - len_patern_sentence):
            string_to_test = text_tokens[i:i + len_patern_sentence]
            string_to_test = ' '.join(string_to_test)
            if string_to_test == patern_sentence:
                data.loc[ind, 'new_answer_start'] = i
                data.loc[ind, 'new_answer_end'] = (i+len_patern_sentence)-1


