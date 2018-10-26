# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.layers.convolutional_recurrent import ConvLSTM2D
import tensorflow as tf
import cv2
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


def stack_frame(i, x_train, frame_size):
    if i >= frame_size:
        return x_train[i - frame_size: i]
    else:
        stack = []
        for j in range(0, i):
            stack.append(x_train[j])
        for j in range(i, frame_size):
            stack.append(x_train[i])
        stack = np.array(stack)
        return stack
        
def read_video(path):
    cap = cv2.VideoCapture(path)
    count = 0
    frames = []
    success = True
    while success:
        try:
            success, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (100, 170))
            #frame = imutils.resize(frame, width=400)
            frame = frame.reshape((1, frame.shape[0], frame.shape[1]))
            frames.append(frame)
            count += 1
        except:
            continue
    return np.array(frames)

def read_data(path):
    train = read_video(path + 'train.mp4')

    speeds = []
    with open(path + 'train_speed.log', 'r') as sfile:
        lines = sfile.readlines()
        for l in lines:
            speeds.append(float(l))
    speeds = np.array(speeds)
    
    test = read_video(path + 'test.mp4')
    
    return train, speeds, test

class Speed:
    def __init__(self, epochs=8, batch_size=9, frame_size=32, lr=0.000005):
        self.epochs = epochs
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.lr = lr
        self.create_model()
        
    def read(self, path):
        self.x_train, self.y_train, self.x_test = read_data(path)
        self.idx = [i for i in range(len(self.x_train))]
        self.test_idx = [i for i in range(len(self.x_test))]

    def add_starting_frames(self):
        first = self.x_test[0]
        minim = float("inf")
        for i in range(len(self.x_train)):
            sum_sq = np.sum((first - i) ** 2)
            if sum_sq < minim:
                minim = sum_sq
                img = i
        self.x_test = np.concatenate((self.x_train[i - self.frame_size:i], self.x_test))
        self.test_idx = [i + self.frame_size for i in self.test_idx]
        
    def shuffle_idx(self):
        idx = list(idx)
        np.random.shuffle(idx)
        
    def get_batch(self, train=True):
        if train:
            while True:
                x = []
                y = []
                for i in range(self.batch_size):
                    try:
                        j = next(self.idx)
                        x.append(stack_frame(j, self.x_train, self.frame_size))
                        y.append(self.y_train[j])
                    except:
                        pass
                x = np.array(x)
                x = np.transpose(x, [0, 1, 3, 4, 2])
                y = np.array(y)
                y = np.reshape(y, (len(y), 1))
                yield x, y
        else:
            while True:
                x = []
                y = []
                try:
                    for i in range(self.batch_size):
                        j = next(self.test_idx)
                        x.append(stack_frame(j, self.x_test, self.frame_size))
                        y.append(1)
                except:
                    pass
                x = np.array(x)
                x = np.transpose(x, [0, 1, 3, 4, 2])
                y = np.array(y)
                y = np.reshape(y, (len(y), 1))
                yield x, y
        
    
    def create_model(self):
        self.model = Sequential()
        self.model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3),
                           input_shape=(self.frame_size, 170, 100, 1),
                           padding='same', return_sequences=True))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        self.model.summary()
        return self.model
        
    def train(self):
        for ep in range(self.epochs):
            self.idx = [i for i in range(len(self.x_train))]
            np.random.shuffle(self.idx)
            self.idx = iter(self.idx)
            batcher = self.get_batch()
            for i in range(np.ceil(len(self.x_train) / self.batch_size):
                if i % 100 == 0:
                    print(i)
                x, y = next(batcher)
                self.model.fit(x, y, verbose=0)
                y_pred = self.model.predict(x)
                mse = np.mean((y - y_pred) ** 2)
            print("Epoch: {}; Train MSE: {}".format(ep + 1, mse))
            self.model.save('model.h5')
            
    def predict(self):
        self.idx = [i for i in range(len(self.x_train))]
        self.idx = iter(self.idx)
        self.test_idx = iter(self.test_idx)
        batcher = self.get_batch()
        predictions = []
        for i in range(np.ceil(len(list(self.test_idx)) / self.batch_size):
            if i % 100 == 0:
                print(i)
            x, y = next(batcher)
            pred = self.model.predict(x)
            predictions.append(pred)
        return np.array(predictions)
        
    def load(self, path):
        self.model = load_model(path)
        
m = Speed()
m.read('../input/speed-detection/')
m.add_starting_frames()
m.train()
#m.load('../input/vehicle-speed-detection/model.h5')
#p = m.predict()
#pf = p.flatten()
#pf[pf < 0] = 0
#with open('test.txt', 'w') as f:
    #for i in pf:
        #f.write(str(i) + '\n')
    
