from nltk import word_tokenize
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical

import json
import logging
import os
import re


log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_format)
path = os.path.abspath(".")

class DataLoading(object):
    def __init__(self,
                config,
                do_load_train_data,
                do_load_test_data,
                eval_data_size=3000,
                path_train_data=os.path.join(path, "data_technical_test/train_technical_test.csv"),
                path_test_data=os.path.join(path, "data_technical_test/test_technical_test.csv"),
                train_vector=False,
                only_with_train_data=False):
        self.config = config
        self.do_load_train_data = do_load_train_data
        self.do_load_test_data = do_load_test_data
        self.eval_data_size = eval_data_size
        self.path_train_data = path_train_data
        self.path_test_data = path_test_data
        self.train_vector = train_vector
        self.only_with_train_data = only_with_train_data

        if self.train_vector:
            self.__train_and_save_encoder()
        else:
            self.__load_encoders()

    def __train_and_save_encoder(self):
        self.df = pd.read_csv(self.path_train_data)
        if not self.only_with_train_data:
            self.test = pd.read_csv(self.path_test_data)
            self.df = pd.concat([self.df, self.test]).reset_index(drop=True)

        self.column_names = self.df.columns.to_list()
        self.__fill_data()
        self.__clean_column()
        self.__categorie_encoder()
        self.__text_vectorizer()

    def __load_encoders(self):
        self.__load_data()
        self.__fill_data()
        self.__clean_column()
        self.categorie_encoder = pickle.load(open(os.path.join(path, 'configuration/categorie_encoder.pkl'), 'rb'))
        logging.debug(f"loaded categorie_encoder succesful")
        self.vectorizer = pickle.load(open(os.path.join(path, 'configuration/vector.pkl'), 'rb'))
        logging.debug(f"loaded vectorizer succesful")

    def __load_data(self):
        if self.do_load_train_data:
            self.df = pd.read_csv(self.path_train_data)
            logging.debug(f"loading train data from {self.path_train_data} succesful")
        elif self.do_load_test_data:
            self.df = pd.read_csv(self.path_test_data)
            logging.debug(f"loading test data from {self.path_test_data} succesful")
        self.column_names = self.df.columns.to_list()
        
    def __fill_data(self):
        for column_name in self.config['column_numerique']:
            self.df[column_name] = self.df[column_name].fillna(0)
        
        for column_name in self.config['column_category'] + self.config['column_with_text']:
            self.df[column_name] = self.df[column_name].fillna('')
        logging.debug(f"filling data succesful")

    def __get_corpus(self):
        corpus = []
        for column_name in self.config['column_with_text']+self.config['column_category']:
            corpus.extend(self.df[column_name].to_list())
        return corpus

    def __text_vectorizer(self):
        corpus = self.__get_corpus()
        vectorizer = CountVectorizer(ngram_range=(self.config['vectorizer_min_ngram'], 
                                    self.config['vectorizer_max_ngram']), 
                                    max_features=self.config['vectorizer_max_features'], 
                                    analyzer='word', 
                                    lowercase=True, 
                                    strip_accents='unicode', 
                                    tokenizer=word_tokenize)
        vectorizer.fit(corpus)
        pickle.dump(vectorizer, open(os.path.join(path, 'configuration/vector.pkl'), 'wb'))
        logging.debug(f"trained vectorizer succesful")

    def __get_categorie(self):
        categories = []
        for column_name in self.config['column_category']:
            categories.extend(self.df[column_name].to_list())
        return categories

    def __categorie_encoder(self):
        categories = self.__get_categorie()
        categorie_encoder = LabelEncoder().fit(categories)
        pickle.dump(categorie_encoder, open(os.path.join(path, 'configuration/categorie_encoder.pkl'), 'wb'))
        logging.debug(f"trained categorie_encoder succesful")
    
    def __clean_column(self):
        logging.debug(f"cleaning data ......")
        map_path = os.path.join(os.path.join(path, 'configuration/', 'char_replace_map.json'))
        with open(map_path, 'r', encoding='UTF-8') as f:
            char_replace_map = json.loads(f.read())
        for key in char_replace_map.keys():
            if key in self.column_names:
                replace_map = char_replace_map[key]
                for i in range(self.df.shape[0]):
                    self.df.loc[i, key] = self.__replacing_multiple_chars(replace_map, self.df.loc[i, key])
        
    def __replacing_multiple_chars(self, char_replace_map, text):
        char_replace_map = dict((re.escape(k), v) for k, v in char_replace_map.items())
        pattern = re.compile("|".join(char_replace_map.keys()))
        result = pattern.sub(lambda m : char_replace_map[re.escape(m.group(0))], text.lower())
        return result

    def categorie_to_numeric(self):
        _array = np.zeros((self.df.shape[0], self.categorie_encoder.classes_.size))
        for column_name in self.config['column_category']:
            encoder = self.categorie_encoder.transform(self.df[column_name].values)
            encoder = to_categorical(np.array([encoder]).T)
            pad = np.zeros((self.df.shape[0], self.categorie_encoder.classes_.size-encoder.shape[1]))
            _array += np.concatenate([encoder, pad], axis=1) # 3682
        self.X = np.concatenate([self.X, _array], axis=1)

    def text_to_numeric(self):
        _array = np.zeros((self.df.shape[0], self.config['vectorizer_max_features']))
        for column_name in self.config['column_with_text']:
            _array += self.vectorizer.transform(self.df[column_name].to_list()).toarray()
        self.X = np.concatenate([self.X, _array], axis=1)

    def __index2label(self):
        self.index2label = {k: v for (k, v) in enumerate(set(self.df[self.config['column_label']].to_list()))}
        self.label2index = {v: k for (k, v) in enumerate(set(self.df[self.config['column_label']].to_list()))}

    def label2array(self):
        if self.config['column_label'] in self.column_names:
            self.__index2label()
            self.index_label = [self.label2index[code] for code in self.df[self.config['column_label']].to_list()]
            self.Y = np.reshape(np.array(self.index_label), (-1, 1))
        else:
            logging.warning(f"don't have label {self.config['column_label']} in data, please add this column")
            self.Y = np.zeros((0,0))

    def transform_x_to_numeric(self):
        self.X = np.array(self.df[self.config['column_numerique']])
        self.categorie_to_numeric()
        self.text_to_numeric()
        self.X = StandardScaler().fit_transform(self.X)  # 39322, 6682 test
        logging.debug(f"transformed data to numeric") 
    
    def transform_y_to_numeric(self):
        self.label2array()
        logging.debug(f"transformed label to numeric")
        
    def train_dev_split(self):
        self.transform_x_to_numeric()
        self.transform_y_to_numeric()

        if self.do_load_train_data:
            x_train, x_dev, y_train, y_dev = train_test_split(self.X, self.Y, test_size = self.eval_data_size, random_state=123)
            return x_train, y_train, x_dev, y_dev
        elif self.do_load_test_data:
            return self.X, self.Y, np.zeros((0, 0)), np.zeros((0, 0))
        

if __name__ == "__main__":
    with open(os.path.join(path, 'configuration/data_loading_config.json')) as config_file:
        config = json.loads(config_file.read())
    data_loading = DataLoading(config=config,
                                do_load_train_data=True,
                                do_load_test_data=False,
                                train_vector=False
                            )
    x_train, y_train, x_dev, y_dev = data_loading.train_dev_split()