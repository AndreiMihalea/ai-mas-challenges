# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

class SqueezenetLSTM(nn.Module):
    def __init__(self, n_layers=2, n_hidden=32, n_output=1):
        super(SqueezenetLSTM, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        inception = models.squeezenet1_1()
        inception.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(*list(inception.children())[:-1])
            
        self.lstm = nn.LSTM(
            input_size = 1000,
            hidden_size = self.n_hidden,
            num_layers = self.n_layers,
            batch_first = True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(n_hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_output)
        )

    def forward(self, x, frames):
        batch_size, timesteps = x.size()[0], x.size()[2]
        weights = self.init_hidden(frames)

        convs = []
        for t in range(timesteps):
            conv = self.conv(x[:, :, t, :, :])
            conv = conv.view(batch_size, -1)
            convs.append(conv)
        convs = torch.stack(convs, 0)
        lstm, _ = self.lstm(convs, weights)
        logit = self.fc(lstm[-1])
        return logit
    
    def init_hidden(self, batch_size):
        hidden_a = torch.randn(self.n_layers, batch_size, self.n_hidden)
        hidden_b = torch.randn(self.n_layers, batch_size, self.n_hidden)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

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
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame[0:310]
            frame = cv2.resize(frame, (224, 224))
            #frame = imutils.resize(frame, width=400)
            frame = frame.reshape((3, frame.shape[0], frame.shape[1]))
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

def get_batch(x_train, y_train, batch_size, frame_size, idx):
    while True:
        x = []
        y = []
        for i in range(batch_size):
            j = next(idx)
            x.append(stack_frame(i, x_train, frame_size))
            y.append(y_train[i])
        x = np.array(x)
        x = np.transpose(x, [0, 2, 1, 3, 4])
        y = np.array(y)
        y = np.reshape(y, (len(y), 1))
        yield x, y
        
def validate(model, x_train, batch_size, frame_size, idx):
    val_batch = get_batch(x_train, y_train, batch_size, frame_size, idx)
    loss_function = nn.MSELoss()
    losses = []
    for j in range(int(len(idx) / batch_size)):
        x, y = next(val_batch)
        x = Variable(torch.from_numpy(x).float())
        y = Variable(torch.from_numpy(y).float())
        model.eval()
        logit = model(x, frame_size)
        loss = loss_function(logit, y)
        losses.append(loss)
        del x, y
        torch.cuda.empty_cache()
    return np.average(losses)
    
def train(model, x_train, y_train, batch_size, frame_size, epochs, learning_rate):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for i in range(epochs):
        model.train()
        split = 1
        # shuffle at the start of the epoch
        train_ids = [i for i in range(0, int(len(x_train) * split))] # train
        #valid_ids = [i for i in range(int(len(x_train) * split), len(x_train))] # valid
        np.random.shuffle(train_ids)
        train_ids = iter(train_ids)
        batch = get_batch(x_train, y_train, batch_size, frame_size, train_ids)
        losses = []
        # iterate through all dataset
        for j in range(int(len(x_train) / batch_size)):
            if j % 100 == 0:
                print(j)
            x, y = next(batch)
            x = Variable(torch.from_numpy(x).float())
            y = Variable(torch.from_numpy(y).float())
            optimizer.zero_grad()
            logit = model(x, frame_size)
            loss = loss_function(logit, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.data[0])
            del x, y
            torch.cuda.empty_cache()
        print("Epoch: {}; Train MSE: {}".format(i, np.average(losses)))
        #val_loss = validate(model, x_train, batch_size, frame_size, valid_ids)
        #print("Epoch: {}; Valid MSE: {}".format(i, np.average(losses)))
    torch.save(model, 'modelll.pt')
        
def predict(model, x_train, x_test, batch_size, frame_size):
    first = x_test[0]
    minim = float("inf")
    for img in range(len(x_train)):
        sum_sq = np.sum((first - img) ** 2)
        if sum_sq < minim:
            minim = sum_sq
            i = img
    x_test = np.concatenate((x_train[i - frame_size:i], x_test))
    test_idx = [i + frame_size for i in range(0, len(x_test))]
    test_ids = iter(test_idx)
    batch = get_batch(model, x_test, np.ones(len(x_test)), batch_size, frame_size, test_ids)
    y_pred = []
    for j in range(int(len(test_idx) / batch_size)):
        if j % 1 == 0:
            print(j)
        x, y = next(batch)
        x = Variable(torch.from_numpy(x).float())
        model.eval()
        logit = model(x, frame_size)
        y_pred.append(logit[0][0])
        del x
        torch.cuda.empty_cache()
    return y_pred
x_train, y_train, x_test = read_data('../input/')
model = SqueezenetLSTM()
#model = torch.load('../input/speed-detection-with-pytorch/model.pt')
#model.eval()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
train(model, x_train, y_train, 16, 32, 5, 0.00001)
'''
y_pred = predict(model, x_train, x_test, 8, 10)
y_pred = y_pred.numpy()
with open('res', 'w') as res:
    for i in y_pred:
        res.write(i + '\n')
'''
