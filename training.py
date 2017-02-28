import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Activation, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle as pk
import os

class DataGen(object):
    def __init__(self, data, ts, batch_sz):
        self.data = data
        self.time_step = ts
        self.batch_sz = batch_sz

    def __iter__(self):
        return self

    def next(self):
        idx = np.random.randint(len(self.data) - self.time_step - 1, size=self.batch_sz)
        X = np.array([self.data[i:i+self.time_step] for i in idx])
        y = np.array([self.data[i+1:i+self.time_step+1] for i in idx])
        return X, y


def get_data(file_path):
    with open(file_path) as f:
        content = "".join(f.readlines()).replace('\r', '')
        c2i = dict([(c, idx) for idx, c in enumerate(set(content))])
        i2c = dict([(c2i[c], c) for c in c2i])
        x = np.zeros((len(content), len(c2i)))
        for idx, c in enumerate(content):
            x[idx, c2i[c]] = 1.0
        return x, i2c


def get_generator(x, time_steps, batch_size):
    return DataGen(x[:len(x)*4/5], time_steps, batch_size), DataGen(x[len(x)*4/5:], time_steps, batch_size)

def get_model(output_dim, input_dim, hidden_sz, dropout_r, optimization):
    model = Sequential()
    model.add(LSTM(hidden_sz, input_dim=input_dim, return_sequences=True))
    if dropout_r:
        model.add(Dropout(dropout_r))
    model.add(TimeDistributed(Dense(output_dim)))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimization, metrics=['acc'])
    return model

if __name__ == '__main__':
    if not os.path.exists('result_model'): os.mkdir('result_model')
    time_steps, batch_size = 25, 100
    hidden_sz = 100
    dropout_rate = 0.0
    optimization = 'rmsprop'
    x, i2c = get_data('data/input.txt')
    model = get_model(len(i2c), len(i2c), hidden_sz, dropout_rate, optimization)
    while True:
        try:
            callbacks = [
                EarlyStopping(monitor='val_acc', min_delta=0.0001, verbose=1, patience=2),
                ModelCheckpoint('result_model/model_t{}_h{}_d{}_o{}'.format(time_steps, hidden_sz, dropout_rate, optimization),
                                monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
            ]
            train, val = get_generator(x, time_steps, batch_size)
            history = model.fit_generator(train, 10000, 100, validation_data=val, nb_val_samples=batch_size, callbacks=callbacks)
            pk.dump(history.history, open('result_model/history_t{}_h{}_d{}_o{}'.format(time_steps, hidden_sz, dropout_rate, optimization), 'wb'))
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            if raw_input("Continue training by increasing time steps? y/n\n".strip()) == 'n': break
            time_steps = int(raw_input('New time step(slightly bigger): \n').strip())
