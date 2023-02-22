from os import walk
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding,Flatten,Dropout
from keras.utils import pad_sequences,to_categorical
from keras.callbacks import ModelCheckpoint,EarlyStopping
import re
import numpy as np

import langid

from pathlib import Path
import pickle

import csv

# Кол-во слов,которые будут выбираться для токенизации
num_words = 30000
# Максимальное кол-во слов в предложениях
max_reviews_len = 250
# Список для отзывов
reviews_list = []
# Разметка отзывов
y_train = []
# Список длин отзывов
df_list = []

reviews_list_test = []
# Разметка отзывов
y_test = []
# Список длин отзывов
df_list_test = []
df_mark = []
df_mark_test = []

# Функция для формирования набора данных
def create_train_data(path,y_data,reviews_list,df_list,df_mark):
    if "train" in path:
        w_file = open("Reviews.csv", mode="a", encoding='utf-8')
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
    else:
        w_file = open("Reviews_test.csv", mode="a", encoding='utf-8')
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
    # Получение всех файлов из папки pos
    for (dirpath, dirnames, filenames) in walk(path):
        print(filenames[:30])
        # Обход всех вайлов
        for name in filenames:
            try:
                with open (f"{path}/{name}","r",encoding="utf-8") as review:
                    extra_list = [0]*10
                    # Добавляем разметку отзыва (превратить в one hot encodin)
                    df_mark.append(int(name[name.find("_")+1:name.find(".")]))
                    extra_list[int(name[name.find("_")+1:name.find(".")])-1] = 1
                    y_data.append(extra_list.copy())
                    # Убираем ненужные символы и приводим текст к нижнему регистру
                    text = review.read().lower()
                    if "<br" in text:
                        text = text.replace("<br"," ")

                    review_text = re.sub(r"[^A-Za-z]"," ",text)
                    reviews_list.append(review_text)
                    extra_list.append(review_text)
                    # Добавляем длину отзыва
                    file_writer.writerow(extra_list.copy())
                    df_list.append(len(review_text.split()))
            except:
                print("name",name)

    return y_data,reviews_list,df_list,df_mark

# Тренировачный набор данных
y_train,reviews_list,df_list,df_mark = create_train_data("F:/GreenAtom/aclImdb/train/pos",y_train,reviews_list,df_list,df_mark)
y_train,reviews_list,df_list,df_mark = create_train_data("F:/GreenAtom/aclImdb/train/neg",y_train,reviews_list,df_list,df_mark)

# Тестовый набор данных
y_test,reviews_list_test,df_list_test,df_mark_test = create_train_data("F:/GreenAtom/aclImdb/test/pos",y_test,reviews_list_test,df_list_test,df_mark_test)
y_test,reviews_list_test,df_list_test,df_mark_test = create_train_data("F:/GreenAtom/aclImdb/test/neg",y_test,reviews_list_test,df_list_test,df_mark_test)

print(len(reviews_list),len(y_train))
print(len(reviews_list_test),len(y_test))
# Функция для вывода квартилей
def check_quartil(df_list):
    df = pd.DataFrame(df_list)
    print(df.describe(percentiles=[.25, .5, .75,.9]))

check_quartil(df_list)
check_quartil(df_mark)
def tokenizer_fit(reviews_list):
    # Задаём параметры токенизации
    tokenizer = Tokenizer(num_words=num_words)

    # Получение числового эквивалента слов
    tokenizer.fit_on_texts(reviews_list)

    return tokenizer

tokenizer = tokenizer_fit(reviews_list)

# Функция для сохранения токенизатора
def save_tokenizer(tz):
    with open('/TokenizerReviews.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

save_tokenizer(tokenizer)

# Функция для вывода пар Числовое представление - Слово
def printtoken(tz):
    n = 1
    for i in tz.word_index:
        if n <= 1000:
            print(f"[ {n} ] - {i}")
            n += 1

# Представление отзывов в числовом виде
def text_to_seq(tz,reviews_list,max_reviews_len):
    # Преобразование текста в числовые последовательности
    reviews_seq = tz.texts_to_sequences(reviews_list)

    # Приведение предложений к определённой длине
    const_reviews_size = pad_sequences(reviews_seq, maxlen= max_reviews_len)

    return const_reviews_size

x_train = text_to_seq(tokenizer,reviews_list,max_reviews_len)
print(x_train[2],y_train[2])
x_test = text_to_seq(tokenizer,reviews_list_test,max_reviews_len)


x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)
val_set = (x_test,y_test)
# Функция для создания модели
def NNmodel(x_train,y_train,max_reviews_len,num_words,val_set):
    model = Sequential()

    model.add(Embedding(input_dim=num_words,output_dim=100,input_length=max_reviews_len))
    model.add(Dropout(0.9))
    model.add(LSTM(63))

    model.add(Dense(10,activation="softmax"))


    model.compile(loss = "categorical_crossentropy",optimizer="adam",metrics = ["accuracy"])
    print(model.summary())
    earlystop = EarlyStopping(monitor="val_accuracy", patience=2)
    callback = ModelCheckpoint("/GA11.h5", monitor="val_accuracy", save_best_only=True,verbose=1)
    model.fit(x_train,y_train,batch_size=32,epochs = 15,validation_data = val_set,callbacks = [earlystop,callback])


NNmodel(x_train,y_train,max_reviews_len,num_words,val_set)
