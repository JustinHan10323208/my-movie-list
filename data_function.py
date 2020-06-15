"""
20200202

@author: j2609

拔掉了GMF跟轉速的輸入
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
# import pydot
# import graphviz
import pickle
import time
import random
import re
from collections import defaultdict
# import math




def numerical(file_path):
        data = pd.read_csv(file_path, sep = '  ', skiprows = [0,1,2], names = ['Ch1','Ch2','Ch3'], engine='python')
                    
        data_Ch1 = np.array(data.Ch1)
        data_Ch2 = np.array(data.Ch2)
        data_Ch3 = np.array(data.Ch3)
        Ch1fft = fft(data_Ch1)
        Ch2fft = fft(data_Ch2)
        Ch3fft = fft(data_Ch3)
        
#        data_Ch1_norm = (data_Ch1 - min(data_Ch1))/(max(data_Ch1)-min(data_Ch1))#做Normalization
#        data_Ch2_norm = (data_Ch2 - min(data_Ch2))/(max(data_Ch2)-min(data_Ch2))
#        data_Ch3_norm = (data_Ch3 - min(data_Ch3))/(max(data_Ch3)-min(data_Ch3))
    
        Ch1fft = Ch1fft[:int(len(Ch1fft)/2)]
        Ch2fft = Ch2fft[:int(len(Ch2fft)/2)]
        Ch3fft = Ch3fft[:int(len(Ch3fft)/2)]
        Ch1fft = abs(Ch1fft)
        Ch2fft = abs(Ch2fft)
        Ch3fft = abs(Ch3fft)
        # Ch1fft_norm = (Ch1fft - min(Ch1fft))/(max(Ch1fft) - min(Ch1fft))
        # Ch2fft_norm = (Ch2fft - min(Ch2fft))/(max(Ch2fft) - min(Ch2fft))
        # Ch3fft_norm = (Ch3fft - min(Ch3fft))/(max(Ch3fft) - min(Ch3fft))
    
        y_ch1 = np.concatenate((data_Ch1, Ch1fft))[:, np.newaxis]
        y_ch2 = np.concatenate((data_Ch2, Ch2fft))[:, np.newaxis]
        y_ch3 = np.concatenate((data_Ch3, Ch3fft))[:, np.newaxis]
        y = np.hstack((y_ch1, y_ch2, y_ch3))[np.newaxis, :, :]
        
        return y
    
def save_history(data, path):
    file = open(path + '.pickle', 'wb')
    pickle.dump(data, file)
    file.close()
    return
def read_history(path):
    with open(path + '.pickle', 'rb') as file:
        history = pickle.load(file)
    return history
def plot_his(his):
    for i in his:
        fig, ax = plt.subplots(2, 1)
        fig.suptitle(str(i))
        acc = np.array(i.history['acc'])[:, np.newaxis]
        acc = np.around(acc*100, decimals = 2)
        epoch = np.arange(len(acc))
        val_acc = np.array(i.history['val_acc'])[:, np.newaxis]
        val_acc = np.around(val_acc*100, decimals = 2)
        acc_total = np.hstack((acc, val_acc))
        loss = np.array(i.history['loss'])[:, np.newaxis]
        val_loss = np.array(i.history['val_loss'])[:, np.newaxis]
        loss_total = np.hstack((loss, val_loss))
        plot0 = ax[0].plot(epoch, acc_total)
        ax[0].legend(plot0, ('train_acc', 'val_acc'))
        ax[0].set_title('Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy(%)')
        ax[0].grid(True)
        plot1 = ax[1].plot(epoch, loss_total)
        ax[1].legend(plot1, ('train_loss', 'val_loss'))
        ax[1].set_title('loss')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel('loss')
        ax[1].grid(True)
        fig.tight_layout()
        

class GearData:
    def __init__(self, total_data_num, array_type):
        self.total_data_num = total_data_num
        self.array_type = array_type
    def read(self, path, train_dict = None, val_dict = None):
        self.data_dict = {}
        self.data_array = np.zeros((1, 15360, 3))
        self.data_label = np.array([])
        for fault_dir in os.listdir(path):
            self.data_dict[fault_dir] = []  #add fault to dictionary
            if self.array_type == 'train' and train_dict == None and val_dict == None:
                #each_number = int(math.ceil(float(self.total_data_num/rot_speed_type)*0.6))
                while len(self.data_dict[fault_dir]) < self.total_data_num: ##Choose data to be added
                    rand_choose =  random.randint(1, len(os.listdir(os.path.join(path, fault_dir))))
                    if rand_choose in self.data_dict[fault_dir]:
                        continue
                    self.data_dict[fault_dir].append(rand_choose)
                    data_name = os.listdir(os.path.join(path, fault_dir))[rand_choose-1]
                    data_path = os.path.join(path, fault_dir, data_name)
                    temp_array = numerical(data_path)
                    self.data_array = np.vstack((self.data_array, temp_array))
                    self.data_label = np.append(self.data_label, fault_dir)
            elif self.array_type == 'val' and train_dict != None and val_dict == None:
                #each_number = int(math.ceil(float(self.total_data_num/rot_speed_type)*0.2))
                while len(self.data_dict[fault_dir]) < self.total_data_num: ##Choose data to be added
                    rand_choose =  random.randint(1, len(os.listdir(os.path.join(path, fault_dir))))  #random choose from all data
                    if rand_choose in self.data_dict[fault_dir] or rand_choose in train_dict[fault_dir]:##if choosed is in dict
                        continue
                    self.data_dict[fault_dir].append(rand_choose)
                    data_name = os.listdir(os.path.join(path, fault_dir))[rand_choose-1]
                    data_path = os.path.join(path, fault_dir, data_name)
                    temp_array = numerical(data_path)
                    self.data_array = np.vstack((self.data_array, temp_array))
                    self.data_label = np.append(self.data_label, fault_dir)
            elif self.array_type == 'test' and train_dict != None and val_dict != None:
                #each_number = int(math.ceil(float(self.total_data_num/rot_speed_type)*0.2))
                while len(self.data_dict[fault_dir]) < self.total_data_num: ##Choose data to be added
                    rand_choose =  random.randint(1, len(os.listdir(os.path.join(path, fault_dir))))
                    if rand_choose in self.data_dict[fault_dir] or rand_choose in train_dict[fault_dir] or rand_choose in val_dict[fault_dir]:
                        continue
                    self.data_dict[fault_dir].append(rand_choose)
                    data_name = os.listdir(os.path.join(path, fault_dir))[rand_choose-1]
                    data_path = os.path.join(path, fault_dir, data_name)
                    temp_array = numerical(data_path)
                    self.data_array = np.vstack((self.data_array, temp_array))
                    self.data_label = np.append(self.data_label, fault_dir)
            else:
                print("Check you arguments!!")
                break
            
        self.data_array = np.delete(self.data_array, 0, axis = 0)
        print("Totally get {0} {1} data.".format(len(self.data_array), self.array_type))
        
    def save_data(self, path, name):
        file = open(os.path.join(path, name + "_array") + '.pickle', 'wb')
        pickle.dump(self.data_array, file)
        file.close()
        file = open(os.path.join(path, name + "_label") + '.pickle', 'wb')
        pickle.dump(self.data_label, file)
        file.close()
        file = open(os.path.join(path, name + "_dict") + '.pickle', 'wb')
        pickle.dump(self.data_dict, file)
        file.close()
        
        
    def load_data(self, file_path, name):
        with open(os.path.join(file_path, name + "_array" + '.pickle'), 'rb') as file:
            self.data_array = pickle.load(file)
        file.close()
        with open(os.path.join(file_path, name + "_label" + '.pickle'), 'rb') as file:
            self.data_label = pickle.load(file)
        file.close()
        with open(os.path.join(file_path, name + "_dict" + '.pickle'), 'rb') as file:
            self.data_dict = pickle.load(file)
        file.close()
        print("Totally get {0} {1} data.".format(len(self.data_array), self.array_type))
        
    def gen_data(self):
        self.data_array_norm = np.copy(self.data_array)
        fft_data = self.data_array[:, 10240:, :]
        short_data = self.data_array[:, :11740, :]
        time_data = self.data_array[:, :10240, :]
        shortfft_data = self.data_array[:, 10240:11740, :]
        self.fft_data = np.copy(fft_data)
        self.short_data = np.copy(short_data)
        self.time_data = np.copy(time_data)
        self.shortfft_data = np.copy(shortfft_data)
        for i in range(len(self.data_array)):          
            self.fft_data[i] = (fft_data[i]-np.amin(fft_data[i], 0))/(np.amax(fft_data[i], 0)- np.amin(fft_data[i], 0))       
            self.time_data[i] = (time_data[i]-np.amin(time_data[i], 0))/(np.amax(time_data[i], 0)- np.amin(time_data[i], 0))
            self.shortfft_data[i] = (shortfft_data[i]-np.amin(shortfft_data[i], 0))/(np.amax(shortfft_data[i], 0)- np.amin(shortfft_data[i], 0))
            self.data_array_norm[i] = np.append(self.time_data[i], self.fft_data[i], axis = 0)
            self.short_data[i] = np.append(self.time_data[i], self.shortfft_data[i], axis = 0)
        
    def label_trans(self):
        self.label1 = np.zeros((1,6))
        self.label_rot = np.zeros((1,1))[:, np.newaxis]
        for i in self.data_label:
            if i[:2] == "Go":
                self.label1 = np.vstack((self.label1, [1, 0, 0, 0, 0, 0]))
            elif i[:2] == "BC":
                self.label1 = np.vstack((self.label1, [0, 1, 0, 0, 0, 0]))
            elif i[:2] == "To":
                self.label1 = np.vstack((self.label1, [0, 0, 1, 0, 0, 0]))
            elif i[:2] == "Lo":
                self.label1 = np.vstack((self.label1, [0, 0, 0, 1, 0, 0]))
            elif i[:2] == "Sh":
                self.label1 = np.vstack((self.label1, [0, 0, 0, 0, 1, 0]))
            else:
                self.label1 = np.vstack((self.label1, [0, 0, 0, 0, 0, 1]))
                
            if re.search(r"1800", i) is not None:
                self.label_rot = np.append(self.label_rot, 1800)
            elif re.search(r"900", i) is not None:
                self.label_rot = np.append(self.label_rot, 900)
            elif re.search(r"2400", i) is not None:
                self.label_rot = np.append(self.label_rot, 2400)
            elif re.search(r"3000", i) is not None:
                self.label_rot = np.append(self.label_rot, 3000)
            else:
                pass
                
        self.label_rot = np.delete(self.label_rot, 0)
        self.label1 = np.delete(self.label1, 0, axis = 0)
                
    def multilabel_trans(self):  #order: Good, BC, Break, Loose, Shaft, Wear
        self.label1 = np.zeros((1,6))
        self.label_rot = np.zeros((1,1))[:, np.newaxis]
        for i in self.data_label:
            if re.search(r"Break", i) is not None:
                if re.search(r"BC", i) is not None:
                    label_t = np.array([0, 1, 1, 0, 0, 0])[np.newaxis, :]
                    self.label1 = np.vstack((self.label1, label_t))
                elif re.search(r"Good", i) is not None:
                    label_t = np.array([1, 0, 1, 0, 0, 0])[np.newaxis, :]
                    self.label1 = np.vstack((self.label1, label_t))
                elif re.search(r"Loose", i) is not None:
                    label_t = np.array([0, 0, 1, 1, 0, 0])[np.newaxis, :]
                    self.label1 = np.vstack((self.label1, label_t))
                elif re.search(r"Shaft", i) is not None:
                    label_t = np.array([0, 0, 1, 0, 1, 0])[np.newaxis, :]
                    self.label1 = np.vstack((self.label1, label_t))
                elif re.search(r"Wear", i) is not None:
                    label_t = np.array([0, 0, 1, 0, 0, 1])[np.newaxis, :]
                    self.label1 = np.vstack((self.label1, label_t))
                else:
                    pass
            elif re.search(r"Loose", i) is not None: #order: Good, BC, Break, Loose, Shaft, Wear
                if re.search(r"BC", i) is not None:
                    label_t = np.array([0, 1, 0, 1, 0, 0])[np.newaxis, :]
                    self.label1 = np.vstack((self.label1, label_t))
                elif re.search(r"Good", i) is not None:
                    label_t = np.array([1, 0, 0, 1, 0, 0])[np.newaxis, :]
                    self.label1 = np.vstack((self.label1, label_t))
                elif re.search(r"Shaft", i) is not None:
                    label_t = np.array([0, 0, 0, 1, 1, 0])[np.newaxis, :]
                    self.label1 = np.vstack((self.label1, label_t))
                elif re.search(r"Wear", i) is not None:
                    label_t = np.array([0, 0, 0, 1, 0, 1])[np.newaxis, :]
                    self.label1 = np.vstack((self.label1, label_t))
                else:
                    pass
            elif re.search(r"Shaft", i) is not None: #order: Good, BC, Break, Loose, Shaft, Wear
                if re.search(r"BC", i) is not None:
                    label_t = np.array([0, 1, 0, 0, 1, 0])[np.newaxis, :]
                    self.label1 = np.vstack((self.label1, label_t))
                elif re.search(r"Good", i) is not None:
                    label_t = np.array([1, 0, 0, 0, 1, 0])[np.newaxis, :]
                    self.label1 = np.vstack((self.label1, label_t))
                elif re.search(r"Wear", i) is not None:
                    label_t = np.array([0, 0, 0, 0, 1, 1])[np.newaxis, :]
                    self.label1 = np.vstack((self.label1, label_t))
                else:
                    pass
            elif re.search(r"Wear", i) is not None: #order: Good, BC, Break, Loose, Shaft, Wear
                if re.search(r"BC", i) is not None:
                    label_t = np.array([0, 1, 0, 0, 0, 1])[np.newaxis, :]
                    self.label1 = np.vstack((self.label1, label_t))
                elif re.search(r"Good", i) is not None:
                    label_t = np.array([1, 0, 0, 0, 0, 1])[np.newaxis, :]
                    self.label1 = np.vstack((self.label1, label_t))
                else:
                    pass
            elif re.search(r"BC", i) is not None: #order: Good, BC, Break, Loose, Shaft, Wear
                if re.search(r"Good", i) is not None:
                    label_t = np.array([1, 1, 0, 0, 0, 0])[np.newaxis, :]
                    self.label1 = np.vstack((self.label1, label_t))
                else:
                    pass
            else:
                pass
            
            if re.search(r"1800", i) is not None:
                self.label_rot = np.append(self.label_rot, 1800)
            elif re.search(r"2400", i) is not None:
                self.label_rot = np.append(self.label_rot, 2400)
            elif re.search(r"3000", i) is not None:
                self.label_rot = np.append(self.label_rot, 3000)
            elif re.search(r"900", i) is not None:
                self.label_rot = np.append(self.label_rot, 900)
            else:
                pass
                
        self.label_rot = np.delete(self.label_rot, 0)
        self.label1 = np.delete(self.label1, 0, axis = 0)
        
    def plot_data(self, num):
        fig, ax = plt.subplots(3, 1)
#        fig.suptitle("Time Domain " + self.label1[num])
        ax[0].set_title(self.label1[num])
        ax[0].plot(np.linspace(0, 1.024, 10240), self.data_array[num, :10240, 0], 'b')
#        ax[0].set_xlabel('Time')
#        ax[0].set_ylabel('X_dir')
        ax[1].plot(np.linspace(0, 1.024, 10240), self.data_array[num, :10240, 1], 'y')
 #       ax[1].set_xlabel('Time')
 #       ax[1].set_ylabel('Y_dir')
        ax[2].plot(np.linspace(0, 1.024, 10240), self.data_array[num, :10240, 2], 'r')
 #       ax[2].set_xlabel('Time')
 #       ax[2].set_ylabel('Z_dir')
        fig1, ax1 = plt.subplots(3, 1)
#        fig1.suptitle("Freq Domain " + self.label1[num])
        ax1[0].plot(np.linspace(0, 5120, 5120), self.data_array[num, 10240:, 0], 'b')
 #       ax1[0].set_xlabel("Freq")
 #       ax1[0].set_ylabel("X_dir")
        ax1[1].plot(np.linspace(0, 5120, 5120), self.data_array[num, 10240:, 1], 'y')
 #       ax1[1].set_xlabel("Freq")
 #       ax1[1].set_ylabel("Y_dir")
        ax1[2].plot(np.linspace(0, 5120, 5120), self.data_array[num, 10240:, 2], 'r')
 #       ax1[2].set_xlabel("Freq")
 #       ax1[2].set_ylabel("Z_dir")
        fig2, ax2 = plt.subplots(3, 1)
        ax2[0].plot(np.linspace(0, 1500, 1500), self.data_array[num, 10240:11740, 0], 'b')
        ax2[1].plot(np.linspace(0, 1500, 1500), self.data_array[num, 10240:11740, 1], 'y')
        ax2[2].plot(np.linspace(0, 1500, 1500), self.data_array[num, 10240:11740, 2], 'r')
        
    
    def build_generator(self, batch_size):
        self.data_generator = MyGenerator(self.data_array_norm, self.label1, batch_size)
        self.shorter_generator = MyGenerator(self.short_data, self.label1, batch_size)
        self.fft_generator = MyGenerator(self.fft_data, self.label1, batch_size)
        self.time_generator = MyGenerator(self.time_data, self.label1, batch_size)
        self.short_fft_generator = MyGenerator(self.shortfft_data, self.label1, batch_size)
    
    def build_test_generator(self, batch_size):
        self.data_generator = MyGenerator_test(self.data_array_norm, batch_size)
        self.shorter_generator = MyGenerator_test(self.short_data, batch_size)
        self.fft_generator = MyGenerator_test(self.fft_data, batch_size)
        self.time_generator = MyGenerator_test(self.time_data, batch_size)
        self.short_fft_generator = MyGenerator_test(self.shortfft_data, batch_size)


class build_net:
    def __init__(self, input_size, net_type, neuron_num):
        self.input_size = input_size
        self.net_type = net_type
        self.neuron_num = neuron_num
        
    def path(self, tf_path, run_times):
        path = self.net_type + str(self.neuron_num)
        self.Tensorboard = tf.keras.callbacks.TensorBoard(log_dir = os.path.join(tf_path, "tensorboard", run_times, path + "-{}".format(time.time())))
        if not os.path.isdir(os.path.join(tf_path, "checkpoint", run_times, path)):
            os.makedirs(os.path.join(tf_path, "checkpoint", run_times, path))
        #cp_path = os.path.join(tf_path, "checkpoint", run_times, path, path + "-{epoch:02d}-acc-{acc:.4f}-val_acc-{val_acc:.4f}.hdf5")
        cp_path = os.path.join(tf_path, "checkpoint", run_times, path,
                               path + ".hdf5")
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(cp_path, monitor='acc', verbose=1, save_best_only=True, mode='max', save_weights_only = True)

    def cnn_net(self, return_model = True):
        inputs = tf.keras.layers.Input(shape = (self.input_size, 3))
        x = tf.keras.layers.Conv1D(16, 5, strides = 1, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(l=0.05))(inputs)
        x = tf.keras.layers.MaxPool1D(4, strides = 4)(x)
        x = tf.keras.layers.Conv1D(32, 51, strides = 1,activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(l=0.05))(x)
        x = tf.keras.layers.MaxPool1D(4, strides = 4)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.neuron_num, activation = 'relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        if return_model:
            x = tf.keras.layers.Dense(6, activation = 'sigmoid')(x)
        else:
            x = tf.keras.layers.Dense(6, activation = 'relu')(x)
        
        self.cnn_model = tf.keras.Model(inputs, x)
        
        if return_model:
            return self.cnn_model
        else:
            pass
        
    
    def mlp_net(self):
        inputs = tf.keras.layers.Input(shape = (2,))
        x = tf.keras.layers.Dense(2, activation = 'relu')(inputs)
        
        self.mlp_model = tf.keras.Model(inputs, x)
    
    def combine_net(self):
        combined = tf.keras.layers.concatenate([self.cnn_model.output, self.mlp_model.output])
        x = tf.keras.layers.Dense(6, activation = 'sigmoid')(combined)
        
        model = tf.keras.Model(inputs = [self.cnn_model.input, self.mlp_model.input], outputs = x)
        return model
    
class MyGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set,y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)

class MyGenerator_test(tf.keras.utils.Sequence):
    def __init__(self, x_set, batch_size):
        self.x = x_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        #return np.array(batch_x1)
        return np.array(batch_x)
                  
## to generate predict and output label from array of number to dict of words
def predict_compare(output, label):  #order: Good, BC, Break, Loose, Shaft, Wear
    error_type = ['Good', 'BC', "Loose", 'Shaft', 'Break', 'Wear']
    predict_single = {}
    predict_comb = {}
    output_single = {}
    output_comb = {}
    predict_single = defaultdict(lambda: [], predict_single)
    predict_comb = defaultdict(lambda: [], predict_comb)
    output_single = defaultdict(lambda: [], output_single)
    output_comb = defaultdict(lambda: [], output_comb)
    
    for index, out, pred in zip(range(len(output)), output, label):
        pred1 = np.where(pred == 1)
        ## predict_single
        for i in pred1[0]: 
            predict_single[str(error_type[i])].append(index)
        ## predict_comb
        if len(pred1[0]) == 1:
            predict_comb[str(error_type[pred1[0][0]])].append(index)
        elif len(pred1[0]) == 2:
            predict_comb[str(error_type[pred1[0][0]])+'+'+str(error_type[pred1[0][1]])].append(index)
        out1 = np.where(out == 1)
        for i in out1[0]: 
            output_single[str(error_type[i])].append(index)
        ## predict_comb
        if len(out1[0]) == 1:
            output_comb[str(error_type[pred1[0][0]])].append(index)
        elif len(pred1[0]) == 2:
            output_comb[str(error_type[pred1[0][0]])+'+'+str(error_type[pred1[0][1]])].append(index)
            
    return predict_single, predict_comb, output_single, output_comb

## to transfer multi-hot to one hot
## order: Good 0, BC 1, Break 2, Loose 3, Shaft 4, Wear 5, GoodBC 6, GoodBreak 7, GoodLoose 8, GoodShaft 9, GoodWear 10,
##        BCBreak 11, BCLoose 12, BCShaft 13, BCWear 14, BreakLoose 15, BreakShaft 16, BreakWear 17, LooseShaft 18, LooseWear 19, ShaftWear 20, Error 21
def to_type(label): 
    temp = []
    for each in label:
        if np.count_nonzero(each == 1) >= 3 or np.count_nonzero(each == 1) == 0:
            temp.append(21)
        else:
            if each[0] == 1:
                if each[1] == 1:
                    temp.append(6)
                elif each[2] == 1:
                    temp.append(7)
                elif each[3] == 1:
                    temp.append(8)
                elif each[4] == 1:
                    temp.append(9)
                elif each[5] == 1:
                    temp.append(10)
                else:
                    temp.append(0)
            elif each[1] == 1:
                if each[2] == 1:
                    temp.append(11)
                elif each[3] == 1:
                    temp.append(12)
                elif each[4] == 1:
                    temp.append(13)
                elif each[5] == 1:
                    temp.append(14)
                else:
                    temp.append(1)
            elif each[2] == 1:
                if each[3] == 1:
                    temp.append(15)
                elif each[4] == 1:
                    temp.append(16)
                elif each[5] == 1:
                    temp.append(17)
                else:
                    temp.append(2)
            elif each[3] == 1:
                if each[4] == 1:
                    temp.append(18)
                elif each[5] == 1:
                    temp.append(19)
                else:
                    temp.append(3)
            elif each[4] == 1:
                if each[5] == 1:
                    temp.append(20)
                else:
                    temp.append(4)
            else:
                temp.append(5)
    return temp



