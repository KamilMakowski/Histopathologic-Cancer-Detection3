from DataService import DataService
from random import shuffle
import cv2
import numpy as np
from keras.layers import Convolution1D, concatenate, SpatialDropout1D, GlobalMaxPool1D, GlobalAvgPool1D, Embedding, \
    Conv2D, SeparableConv1D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU, Flatten
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, \
    Lambda, Multiply, LSTM, Bidirectional, PReLU, MaxPooling1D
from keras.applications.nasnet import NASNetMobile
from keras.applications.nasnet import preprocess_input as preprocess_input_NASNetMobile
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd

class KerasModels:

    def __init__(self, checkpoint_path, data_service, batch_size):
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.train_set = data_service.train_set
        self.test_set = data_service.test_set
        self.labels = data_service.labels
        self.data_service = data_service

    def data_gen_for_NASNetMobile(self, list_files, id_label_map, batch_size, augment=False):
        seq = self.data_service.data_agumentation()
        while True:
            shuffle(list_files)
            for batch in self.data_service.chunker(list_files, batch_size):
                X = [cv2.imread(x) for x in batch]
                Y = [id_label_map[self.data_service.get_id_from_file_path(x)] for x in batch]
                if augment:
                    X = seq.augment_images(X)
                X = [preprocess_input_NASNetMobile(x) for x in X]

                yield np.array(X), np.array(Y)

    def build_NASNetMobile(self, optimizer, lr, weights = None):
        inputs = Input((96, 96, 3))
        input_tensor = Input(shape=(96,96,3))
        base_model = NASNetMobile(include_top=False, input_tensor=input_tensor, weights = weights)#, weights=None
        x = base_model(inputs)
        out1 = GlobalMaxPooling2D()(x)
        out2 = GlobalAveragePooling2D()(x)
        out3 = Flatten()(x)
        out = Concatenate(axis=-1)([out1, out2, out3])
        out = Dropout(0.5)(out)
        out = Dense(1, activation="sigmoid", name="3_")(out)
        model = Model(inputs, out)
        model.compile(optimizer=get_optimizer(optimizer, lr), loss=binary_crossentropy, metrics=['acc'])
        model.summary()

        return model
    
    def get_optimizer(self, optimizer, lr):
        if optimazer == "adam":
            return Adam(lr)
        elif optimazer == "sgd":
            return SGD(lr)
        else:
            raise ValueError(optimizer + " is not supported")

    def train_NASNetMobile(self, model, epochs, augment=True):
        model = self.build_NASNetMobile()
        checkpoint = ModelCheckpoint(self.checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
        self.history = model.fit_generator(
            self.data_gen_for_NASNetMobile(self.train_set, self.labels, self.batch_size, augment=True),
            validation_data=self.data_gen_for_NASNetMobile(self.test_set, self.labels, self.batch_size),
            epochs=epochs, verbose=1,
            callbacks=[checkpoint],
            steps_per_epoch=len(self.train_set) // self.batch_size,
            validation_steps=len(self.test_set) // self.batch_size)

    def show_final_history(self, history=None):
        if history is None:
            history = self.history

        fig, ax = plt.subplots(1, 2, figsize=(15,5))
        ax[0].set_title('loss')
        ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
        ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
        ax[1].set_title('acc')
        ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
        ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
        ax[0].legend()
        ax[1].legend()
        plt.show()

    def save_predict_to_csv_NASNetMobile(self, model, model_weights_math, val_path, csv_path):
        model.load_weights(model_weights_math)
        preds = []
        ids = []
        for batch in self.data_service.chunker(val_path, self.batch_size):
            X = [preprocess_input_NASNetMobile(cv2.imread(x)) for x in batch]
            ids_batch = [self.data_service.get_id_from_file_path(x) for x in batch]
            X = np.array(X)
            preds_batch = ((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*model.predict(X[:, ::-1, ::-1, :]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()
            preds += preds_batch
            ids += ids_batch

        df = pd.DataFrame({'id':ids, 'label':preds})
        df.to_csv(csv_path, index=False)
        df.head()