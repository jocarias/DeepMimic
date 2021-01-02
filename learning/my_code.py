import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import pycwt
import os
import json
import pandas
import tensorflow as tf


class Mode(Enum):
    DEFAULT = 1         # not working
    REDUCED_STATE = 2
    CWT_CNN_v1 = 3
    CWT_CNN_v2 = 4      # not implemented
    CNN_v1 = 5          # not implemented

class Settings(object):
    def mode():
        return Mode.REDUCED_STATE

    # use Bay Walker at the beginning of the training?
    def use_babe_support():
        return False

    #def is_state_vars_set_to_zero():
    #    return False


class MyRawStateData(object):
    def __init__(self, data_size, sample_size):
        self.filename = "rawStateData.npy"
        self.sample_size = sample_size
        self.sample_index = 0
        self.data = np.empty([sample_size, data_size])
        return
    
    def save_state(self, state_info):
        if self.sample_index < self.sample_size:
            self.data[self.sample_index] = state_info
            self.sample_index += 1
            if self.sample_index == self.sample_size:
                np.save(self.filename, self.data)
        return

    def get_data(self):
        return np.load(self.filename)


class Mocap(object):
    def __init__(self):
        MOCAP_FOLDER = "\\data\\motions\\"
        MOCAP_FILE = "humanoid3d_run.txt"
        mocap_file_path = os.getcwd() + MOCAP_FOLDER + MOCAP_FILE
        motion_tag = "Frames"
        column_names = ["duration", "root_px", "root_py", "root_pz", "root_w", "root_x", "root_y", "root_z",
                        "chest_w", "chest_x", "chest_y", "chest_z", "neck_w", "neck_x", "neck_y", "neck_z",
                        "r_hip_w", "r_hip_x", "r_hip_y", "r_hip_z", "r_knee_rot",
                        "r_ankle_w", "r_ankle_x", "r_ankle_y", "r_ankle_z", "r_shoulder_w", "r_shoulder_x", 
                        "r_shoulder_y", "r_shoulder_z", "r_elbow_rot", "l_hip_w", "l_hip_x", "l_hip_y", "l_hip_z",
                        "r_knee_rot", "l_ankle_w", "l_ankle_x", "l_ankle_y", "l_ankle_z",
                        "r_shoulder_w", "r_shoulder_x", "r_shoulder_y", "r_shoulder_z", "l_elbow_rot"]
        columns_to_drop = ["duration", "root_px", "root_py", "root_pz", "root_w", "root_x", "root_y", "root_z",
                           "chest_w", "chest_x", "chest_y", "chest_z", "neck_w", "neck_x", "neck_y", "neck_z", 
                           "r_ankle_w", "r_ankle_x", "r_ankle_y", "r_ankle_z",
                           "l_ankle_w", "l_ankle_x", "l_ankle_y", "l_ankle_z"]

        dataframe = pandas.DataFrame(data=self.__get_data_from_json(mocap_file_path, motion_tag),columns=column_names)
        dataframe.drop(columns=columns_to_drop, inplace=True)
        self.column_names = dataframe.columns.to_numpy()
        self.data = dataframe.to_numpy()
    
    def __get_data_from_json(self, file_path, tag):
        data_file = open(file_path, "r")
        data = json.load(data_file)[tag]
        data_file.close()
        return data


class MyPlots(object):
    def show_signal(signal):
        plt.figure(figsize=(18, 6))
        plt.plot(signal)
        plt.show()

    def show_scalogram(channel_name, coefs, signal, coi, scale_max, scale_min):
        signal_lenght = len(signal)
        fig = plt.figure(figsize=(12, 10))
        fig.subplots_adjust(hspace=0.3, wspace=0.1, bottom=.03, left=.05, right=.99, top=.98)
        gs = fig.add_gridspec(4,1)
        ax_samples = fig.add_subplot(gs[0,0])
        ax_lin = fig.add_subplot(gs[1:,0])
        ax_samples.margins(0, 0.02)
        ax_samples.set_title(channel_name)
        ax_samples.plot(signal)
        # plt.cm.seismic -> blue: low values, red: high values.
        cmap = plt.cm.seismic
        
        ax_lin.imshow(coefs, extent=[0, signal_lenght, scale_max, scale_min], cmap=cmap, aspect='auto',
                                    vmax=abs(coefs).max(), vmin=-abs(coefs).max())
        if coi is not None:
            plt.plot(np.arange(signal_lenght), coi)
        plt.show()


class MyMemoryBuffer(object):
    def __init__(self, channel_array, buffer_length, cache_size):
        self.channels = channel_array
        self.channel_count = len(channel_array)
        self.length = buffer_length
        self.cache_size = cache_size
        self.__buffer = np.zeros((self.channel_count, self.length + self.cache_size))
        self.curr_cache_index = 0

    # updates the buffer so it contains the latest data
    def __checkout(self):
        if self.cache_size == self.curr_cache_index:
            self.__buffer[:,:-self.cache_size] = self.__buffer[:,self.curr_cache_index:]
        elif self.curr_cache_index > 0:
            self.__buffer[:,:-self.cache_size] = self.__buffer[:,self.curr_cache_index:-self.cache_size+self.curr_cache_index]
        self.curr_cache_index = 0

    # returns only valid memory
    def get_buffer(self):
        self.__checkout()
        return self.__buffer[:, :-self.cache_size]

    # saves data to hidden section
    def save(self, state_info):
        if self.curr_cache_index < self.cache_size:
            self.__buffer[:,-self.cache_size+self.curr_cache_index] = state_info
            self.curr_cache_index += 1
        else:
            self.__buffer[:,:-1] = self.__buffer[:,1:]
            self.__buffer[:,-1] = state_info
    
    def reset(self):
        self.__buffer[:,:] = 0


class MyWT(object):
    def __init__(self, scale_min, scale_max, scale_count):
        self.wavelet_type = "mexicanhat"
        self.order = "freq"  # other option is "normal"
        self.scale_count = scale_count
        self.scales = np.linspace(scale_min, scale_max+1, num=scale_count, endpoint=False, dtype=int)
        self.freqs = 1/(pycwt.MexicanHat().flambda() * self.scales)
        self.delta_t = 1

    def __normalize(self, signal):
        signal_min = signal.min()
        signal_max = signal.max() - signal_min
        if signal_max != 0:
            return (2 * (signal - signal_min) / signal_max) - 1
        return signal

    def calculate_cwt(self, memory_buffer, return_plot_info=False):
        coefs_array = []
        if not return_plot_info:
            row_count = self.scale_count
            col_count = memory_buffer.length
            coefs_array = np.ndarray([row_count, col_count, memory_buffer.channel_count])
            channel_index = 0
        
        buffer = memory_buffer.get_buffer()
        for idx, x in enumerate(buffer):
            channel_name = memory_buffer.channels[idx]
            x = self.__normalize(x)
            coefs, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(x, self.delta_t, wavelet=self.wavelet_type, freqs=self.freqs)
            coefs = self.__normalize(coefs.real)

            # "clean" values away from the max/min
            #coefs = np.power(coefs, 3) 

            if not return_plot_info:
                for i in range(row_count):
                    for j in range(col_count):
                        coefs_array[i][j][channel_index] = coefs[i][j]
                channel_index += 1
            else:
                coefs_array.append((channel_name, coefs, x, coi))

        return coefs_array


class MyCNN(object):
    NAME = "my_cnn"

    def __init__(self, input_tfs, channel_count):
        self.last_layer_output_size = 128
        self.network = self.__build_net(input_tfs, channel_count)

    def __build_net(self, input_tfs, channel_count):
        filter_count1 = 16
        filter_count2 = 32
        new_cnn = None
        with tf.variable_scope("cnn"):
            w1 = tf.get_variable("w1",[3,3,channel_count,filter_count1], initializer=tf.random_normal_initializer())
            b1 = tf.get_variable("b1", [filter_count1], initializer=tf.random_normal_initializer())
            new_cnn = tf.nn.conv2d(input_tfs, w1, strides=[1, 1, 1, 1], padding="SAME")
            new_cnn = tf.nn.relu(new_cnn + b1)
            new_cnn = tf.nn.max_pool(new_cnn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            
            w2 = tf.get_variable("w2",[3,3,filter_count1,filter_count2], initializer=tf.random_normal_initializer())
            b2 = tf.get_variable("b2", [filter_count2], initializer=tf.random_normal_initializer())
            new_cnn = tf.nn.conv2d(new_cnn, w2, strides=[1, 1, 1, 1], padding="SAME")
            new_cnn = tf.nn.relu(new_cnn + b2)
            new_cnn = tf.nn.max_pool(new_cnn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            
            last_layer_input_size = new_cnn.shape[1].value * new_cnn.shape[2].value * new_cnn.shape[3].value
            new_cnn = tf.reshape(new_cnn, [-1, last_layer_input_size])
            w3 = tf.get_variable("w3",[last_layer_input_size, self.last_layer_output_size], initializer=tf.random_normal_initializer())
            b3 = tf.get_variable("b3", [self.last_layer_output_size], initializer=tf.random_normal_initializer())

            new_cnn = tf.nn.sigmoid(tf.matmul(new_cnn, w3) + b3)
        return new_cnn
