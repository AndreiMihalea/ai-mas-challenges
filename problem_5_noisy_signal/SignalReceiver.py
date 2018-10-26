import argparse
from collections import deque
import numpy as np
import scipy
import pandas as pd
from scipy import signal

parser = argparse.ArgumentParser(description='Test number')
parser.add_argument('test_no', metavar='N', type=int, help='Test number.')


class SignalReceiver:
	def __init__(self, test_no):
		assert 1 <= test_no <= 5, "Invalid test number."
		f_real = open('Tests/Test' + str(test_no) + '_real')
		f_noisy = open('Tests/Test' + str(test_no))

		self.__noisy_values = [float(i) for i in f_noisy.readlines()]
		self.__real_values = [float(i) for i in f_real.readlines()]
		self.__c_index = 0
		self.__total_error = 0

	def get_value(self):
		'''
		Gets next noisy value from device. This must be called before push_value.
		:return: (float) device value, None if the device is closed.
		'''
		if self.__c_index >= len(self.__noisy_values):
			return None

		val = self.__noisy_values[self.__c_index]
		self.__c_index = self.__c_index + 1
		return val

	def push_value(self, c_val):
		'''
		Computes the error between the real signal and the corrected value.
		:param c_val: corrected value.
		:return: (float) error value, None if the device is closed.
		'''

		if self.__c_index - 1 >= len(self.__real_values):
			return None

		error = abs(self.__real_values[self.__c_index - 1] - c_val)
		self.__total_error += error
		return error

	def get_error(self):
		return self.__total_error

def compute_value(history, value):
	diff = history[-1] - value
	diff_window = history[-20] - history[-1]
	if np.abs(diff) > np.abs(diff_window) * 3:
		value = history[-1] + diff_window / 20
	return value

def c_value(history):
	y = signal.savgol_filter(history, 61, 3)
	return y[-1]

if __name__ == "__main__":
	'''
	Dumb example of usage.
	'''
	args = parser.parse_args()
	sr = SignalReceiver(args.test_no)
	
	n = 9
	b = [1.0 / n] * n
	a = 1

	i_val = sr.get_value()
	history = [i_val]
	pl = []

	count = 0

	while i_val:
		#push_val = lfilter(b, a, history + [i_val])[-1]
		if count > 61:
			push_val = compute_value(history[:-1], i_val)
			#push_val = c_value(history[:-1] + [push_val])
		elif count > 21:
			push_val = compute_value(history[:-1], i_val)
		else:
			push_val = i_val
		print(sr.push_value(push_val))
		pl.append(push_val)
		history.append(i_val)
		i_val = sr.get_value()
		count += 1
	print('Total error: ' + str(sr.get_error()))
	import matplotlib.pyplot as plt
	plt.plot([i for i in range(len(pl))], pl)
	plt.show()
