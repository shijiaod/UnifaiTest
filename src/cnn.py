import argparse
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

import json
import logging
import os

from loading_data import DataLoading


log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_format)
path = os.path.abspath(".")

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", os.path.join(path, 'model/cnn/'),
                    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("train_data", "data_technical_test/train_technical_test.csv", "train_data_filename")
flags.DEFINE_string("test_data", "data_technical_test/test_technical_test.csv", "test_data_filename")
flags.DEFINE_string("checkpoint_path", "cp-{epoch:04d}.ckpt", "The model name for saving")

flags.DEFINE_bool("do_train", True, "if train")
flags.DEFINE_bool("do_eval", True, "if evaluate")
flags.DEFINE_bool("do_test", False, "if test")
flags.DEFINE_bool("save_best_only", True, "if test")
flags.DEFINE_integer("num_classes", 1468, "nomber of class to predict")
flags.DEFINE_integer("epochs", 10, "max iter")
flags.DEFINE_integer("batch_size", 64, "batch_size")
flags.DEFINE_integer("eval_data_size", 5000, "size of data to split from train data for evaluate")


class CNN(object):
    def __init__(self):
        self.model = self.__create_model()
        self.__data_loading()
        if FLAGS.do_test:
            self.load_model()

    def __create_model(self):
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(6682, 1)))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu')) 
        model.add(MaxPooling1D(pool_size=3 ))
        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(3000))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(FLAGS.num_classes))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))
        model.compile(
            'adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model
    
    def __data_loading(self):
        with open(os.path.join(path, 'configuration/data_loading_config.json'), 'r', encoding='UTF-8') as f:
            data_loading_config = json.loads(f.read())
        self.data_loading = DataLoading(
            config = data_loading_config,
            do_load_train_data = FLAGS.do_train,
            do_load_test_data = FLAGS.do_test,
            eval_data_size = FLAGS.eval_data_size,
            path_train_data = os.path.join(path, FLAGS.train_data),
            path_test_data = os.path.join(path, FLAGS.test_data)
        )

    def __reshape(self, x, y):
        x = x.reshape(x.shape[0], x.shape[1], 1).astype('float32')
        y = to_categorical(y, FLAGS.num_classes)
        return x, y


    def load_data(self):
        x_train, y_train, x_dev, y_dev = self.data_loading.train_dev_split()
        x_train, y_train = self.__reshape(x_train, y_train)
        x_dev, y_dev = self.__reshape(x_dev, y_dev)
        logging.debug(f"loaded data succesful")
        return x_train, y_train, x_dev, y_dev
    
    def train(self, x_train, y_train, x_dev, y_dev):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(FLAGS.output_dir, FLAGS.checkpoint_path),
            save_weights_only=True,
            monitor='val_acc', 
            verbose=1,
            save_best_only=FLAGS.save_best_only,
            mode='max')
        self.model.fit(x_train, 
                        y_train, 
                        epochs=FLAGS.epochs, 
                        batch_size=FLAGS.batch_size, 
                        verbose=1, 
                        callbacks=[cp_callback])
        if FLAGS.do_eval:
            self.evaluate(x_dev, y_dev)
    
    def load_model(self):
        latest = tf.train.latest_checkpoint(FLAGS.output_dir)
        self.model.load_weights(latest).expect_partial()
        logging.debug(f"loaded model from {latest} succesful")
        
    def evaluate(self, x_test, y_test):
        if FLAGS.do_test:
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1).astype('float32')
        if y_test.shape[0] > 0:
            logging.debug(f"shape of x_test and y_test are {x_test.shape}, {y_test.shape}")
            loss, acc = self.model.evaluate(x_test, y_test, verbose=2)
            logging.info(f"====== EVALUATION ======")
            logging.info(f"Restored model, accuracy: {acc}, loss : {loss}")
        else:
            logging.warning(f"y_test.shape[0] is 0: {y_test}")


def main():
    if FLAGS.do_train:
        model = CNN()
        x_train, y_train, x_dev, y_dev = model.load_data()
        model.train(x_train, y_train, x_dev, y_dev)
    if FLAGS.do_test:
        model = CNN()
        x_test, y_test, _, _ = model.load_data()
        model.evaluate(x_test, y_test)
        

if __name__ == "__main__":
    main()
