# The speed challenge

Your challenge is to predict the speed of the **Nemodrive** car from a video.

_Inspiration from https://github.com/commaai/speedchallenge._

Surprise us with your "speed" skill!

## Getting Started

You can use whatever tools or algorithms you want.

One suggestion would be using python with tools like opencv (to read the data) and pytorch (to 
train a model). Also can use classical vision algorithms to solve this problem? Let's find the 
best solution.

Learning resource: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

## Submission Format

Send us your speed predictions for the test video (`test.mp4`). We will evaluate your test.txt using mean squared error.

The test file will be disclosed the latest on Monday, the 15th of October.

**Send us the test.txt file.** 

## Data overview

Videos are shot at 10 fps with a resolution of 640x360. 

You can download the data from the following link: https://goo.gl/eNSEZy.

* `speed_data/train.mp4` is a video of driving around ACS Building containing 14400 frames.
* `speed_data/train_speed.log` contains the speed of the car at each frame in km/h, one speed on each 
line.
* `speed_data/test.mp4` is the test dataset containing 6500 frames.

## Solution
Used the Kaggle GPU instances for training
 * Keras: the solution that outputed the test.txt file
 Uses a ConvLSTM neural netowrk and builds a stack of frames for each timestep of the video. It contains a sequence of past frames, including the current one. For the prediction, I searched for the most similar image from the train and appended a sequence of equal length with the frame size at the beginning of the test set.
 * PyTorch: not the full solution, because I was getting out of memory errors for the prediction.
 Uses a squeezenet convolutional neural network, then the flatten input is fed to a LSTM net.
