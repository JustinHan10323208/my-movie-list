'''
change code to tf2.1
'''

import data_function
from data_function import to_type
import tensorflow as tf
import os
import random
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

#for i in range(10):
#    np.random.seed(i)
#    random.seed(i)
#    tf.random.set_seed(i)


#date = "20200305"
comb_path = r"D:\Lab\Gear Comb"
single_path = r"D:\Lab\Gear Data new"
dir_path = os.path.join("D:\\TFcode\\Final\\20200615", "data")
tf_path = r"D:\\TFcode\\Final\\20200615" + "\\"

##each data number is 1000, so each train 1000*0.6 = 600, val & test = 1000*0.2 ~= 200
##Each have data_array, fft_data, short_data, time_data, data_label, data_dict, label1(multilabel), label_rot

train_comb = data_function.GearData(350, 'train')
#train_comb.read(comb_path)
#train_comb.save_data(dir_path, "train_comb")
# train_comb_array1 = np.copy(train_comb.data_array[:7875])
# train_comb_array2 = np.copy(train_comb.data_array[7875:])
# file = open(os.path.join(dir_path, "train_comb_array1") + '.pickle', 'wb')
# pickle.dump(train_comb_array1, file)
# file.close
# file = open(os.path.join(dir_path, "train_comb_array2") + '.pickle', 'wb')
# pickle.dump(train_comb_array2, file)
# file.close
# file = open(os.path.join(dir_path, "train_comb_label") + '.pickle', 'wb')
# pickle.dump(train_comb.data_label, file)
# file.close
# file = open(os.path.join(dir_path, "train_comb_dict") + '.pickle', 'wb')
# pickle.dump(train_comb.data_dict, file)
# file.close
## load_data
# with open(os.path.join(dir_path, 'train_comb' + "_array1" + '.pickle'), 'rb') as file:
#     train_comb_array1 = pickle.load(file)
# file.close()
# with open(os.path.join(dir_path, 'train_comb' + "_array2" + '.pickle'), 'rb') as file:
#     train_comb_array2 = pickle.load(file)
# file.close()
# with open(os.path.join(dir_path, 'train_comb' + "_dict" + '.pickle'), 'rb') as file:
#     train_comb.data_dict = pickle.load(file)
# file.close()
# with open(os.path.join(dir_path, 'train_comb' + "_label" + '.pickle'), 'rb') as file:
#     train_comb.data_label = pickle.load(file)
# file.close()
train_comb.data_array = np.vstack((train_comb_array1, train_comb_array2))
#train_comb.load_data(dir_path, "train_comb")
train_comb.gen_data()
train_comb.multilabel_trans()
del train_comb_array1
del train_comb_array2

##train single
train_single = data_function.GearData(350, 'train')
# train_single.read(single_path)
# train_single.save_data(dir_path, "train_single")
train_single.load_data(dir_path, "train_single")
train_single.gen_data()
train_single.label_trans()

##combine two train
train = data_function.GearData(350, 'train')
train.data_array = np.vstack((train_comb.data_array, train_single.data_array))
train.short_data = np.vstack((train_comb.short_data, train_single.short_data))
train.fft_data = np.vstack((train_comb.fft_data, train_single.fft_data))
train.time_data = np.vstack((train_comb.time_data, train_single.time_data))
train.data_array_norm = np.vstack((train_comb.data_array_norm, train_single.data_array_norm))
train.shortfft_data = np.vstack((train_comb.shortfft_data, train_single.shortfft_data))
train.label1 = np.vstack((train_comb.label1, train_single.label1))
#train.label_rot = np.append(train_comb.label_rot, train_single.label_rot)
##build 4 generator, data_generator, shorter_generator, fft_generator, time_generator
train.build_generator(50)



#val comb
val_comb = data_function.GearData(150, 'val')
# val_comb.read(comb_path, train_comb.data_dict)
# val_comb.save_data(dir_path, "val_comb")
val_comb.load_data(dir_path, "val_comb")
val_comb.gen_data()
val_comb.multilabel_trans()

#val single
val_single = data_function.GearData(150, 'val')
#val_single.read(single_path, train_single.data_dict)
#val_single.save_data(dir_path, "val_single")
val_single.load_data(dir_path, "val_single")
val_single.gen_data()
val_single.label_trans()

#combine two val
val = data_function.GearData(150, 'val')
val.data_array = np.vstack((val_comb.data_array, val_single.data_array))
val.short_data = np.vstack((val_comb.short_data, val_single.short_data))
val.fft_data = np.vstack((val_comb.fft_data, val_single.fft_data))
val.shortfft_data = np.vstack((val_comb.shortfft_data, val_single.shortfft_data))
val.time_data = np.vstack((val_comb.time_data, val_single.time_data))
val.data_array_norm = np.vstack((val_comb.data_array_norm, val_single.data_array_norm))
val.label1 = np.vstack((val_comb.label1, val_single.label1))
#val.label_rot = np.append(val_comb.label_rot, val_single.label_rot)
##build 4 generator, data_generator, shorter_generator, fft_generator, time_generator
#del val_comb
#del val_single
val.build_generator(50)


del train_comb
del train_single
del val_comb
del val_single

    #test_comb
    # test_comb = data_function.GearData(67, 'test')
    # test_comb.read(comb_path, train_comb.data_dict, val_comb.data_dict)
    # test_comb.save_data(dir_path, 'test_comb')
    # #test_comb.load_data(dir_path, 'test_comb')
    # test_comb.gen_data()
    # test_comb.multilabel_trans()

    # #test single
    # test_single = data_function.GearData(67, 'test')
    # test_single.read(single_path, train_single.data_dict, val_single.data_dict)
    # test_single.save_data(dir_path, 'test_single')
    # #test_single.load_data(dir_path, 'test_single')
    # test_single.gen_data()
    # test_single.label_trans()

    # #combine two test
    # test = data_function.GearData(67, 'test')
    # test.data_array = np.vstack((test_comb.data_array, test_single.data_array))
    # test.short_data = np.vstack((test_comb.short_data, test_single.short_data))
    # test.fft_data = np.vstack((test_comb.fft_data, test_single.fft_data))
    # test.shortfft_data = np.vstack((test_comb.shortfft_data, test_single.shortfft_data))
    # test.time_data = np.vstack((test_comb.time_data, test_single.time_data))
    # test.label1 = np.vstack((test_comb.label1, test_single.label1))
    # test.build_test_generator(21)
#    import matplotlib.pyplot as plt
#    fig, ax = plt.subplots(3,1, sharex = True)
#    ax.(fontsize = 20)
##    ax[0].plot(np.arange(len(val.shortfft_data[5,:,0])), val.shortfft_data[5,:,0], color = 'b')
#    ax[1].plot(np.arange(len(val.shortfft_data[5,:,1])), val.shortfft_data[5,:,1], color = 'y')
#    ax[2].plot(np.arange(len(val.shortfft_data[5,:,2])), val.shortfft_data[5,:,2], color = 'r')



sgd = tf.keras.optimizers.Adam(learning_rate = 0.001)

class early_stop(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_acc')>= 0.999):
            self.model.stop_training = True

ES = early_stop()
ES1 = tf.keras.callbacks.EarlyStopping('acc', 0.0001, 5)

dense_list = [40, 80, 120, 160, 200, 240]
for run_time in range(0,10):
    tf.random.set_seed(run_time)
    run_time = str(run_time)
    normal = {}
    normal_his = {}
    normal_model = {}
    shorter = {}
    shorter_his = {}
    shorter_model = {}
    fft = {}
    fft_his = {}
    fft_model = {}
    short_fft = {}
    short_fft_his = {}
    short_fft_model = {}
    time1 = {} ##################   time model!!!!!!!!!!!!!!!!!!
    time_his = {}
    time_model = {}
    for neu_num in dense_list:
        normal[str(neu_num)] = data_function.build_net(15360, 'normal', neu_num)
        normal_model[str(neu_num)] = normal[str(neu_num)].cnn_net(return_model = True)
        normal[str(neu_num)].path(tf_path, run_time)
        normal_model[str(neu_num)].compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['acc', 'mse'])
        normal_his[str(neu_num)] = normal_model[str(neu_num)].fit_generator(train.data_generator, steps_per_epoch = 441, epochs = 200,
                                                                      validation_data = val.data_generator, validation_steps = 189,
                                                                      callbacks = [normal[str(neu_num)].Tensorboard, normal[str(neu_num)].checkpoint, ES, ES1])
    for neu_num in dense_list:
        shorter[str(neu_num)] = data_function.build_net(11740, 'shorter', neu_num)
        shorter_model[str(neu_num)] = shorter[str(neu_num)].cnn_net(return_model = True)
        shorter[str(neu_num)].path(tf_path, run_time)
        shorter_model[str(neu_num)].compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['acc', 'mse'])
        shorter_his[str(neu_num)] = shorter_model[str(neu_num)].fit_generator(train.shorter_generator, steps_per_epoch = 441, epochs = 200,
                                                                      validation_data = val.shorter_generator, validation_steps = 189,
                                                                      callbacks = [shorter[str(neu_num)].Tensorboard, shorter[str(neu_num)].checkpoint, ES, ES1])
    # for neu_num in dense_list:
    #     fft[str(neu_num)] = data_function.build_net(5120, 'fft', neu_num)
    #     fft_model[str(neu_num)] = fft[str(neu_num)].cnn_net(return_model = True)
    #     fft[str(neu_num)].path(tf_path, run_time)
    #     fft_model[str(neu_num)].compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['acc', 'mse'])
    #     fft_his[str(neu_num)] = fft_model[str(neu_num)].fit_generator(train.fft_generator, steps_per_epoch = 441, epochs = 200,
    #                                                                   validation_data = val.fft_generator, validation_steps = 189,
    #                                                                   callbacks = [fft[str(neu_num)].Tensorboard, fft[str(neu_num)].checkpoint, ES, ES1])
    # for neu_num in dense_list:
    #     short_fft[str(neu_num)] = data_function.build_net(1500, 'short_fft', neu_num)
    #     short_fft_model[str(neu_num)] = short_fft[str(neu_num)].cnn_net(return_model = True)
    #     short_fft[str(neu_num)].path(tf_path, run_time)
    #     short_fft_model[str(neu_num)].compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['acc', 'mse'])
    #     short_fft_his[str(neu_num)] = short_fft_model[str(neu_num)].fit_generator(train.short_fft_generator, steps_per_epoch = 441, epochs = 200,
    #                                                                   validation_data = val.short_fft_generator, validation_steps = 189,
    #                                                                   callbacks = [short_fft[str(neu_num)].Tensorboard, short_fft[str(neu_num)].checkpoint, ES, ES1])
    # for neu_num in dense_list:
    #     time1[str(neu_num)] = data_function.build_net(10240, 'time', neu_num)
    #     time_model[str(neu_num)] = time1[str(neu_num)].cnn_net(return_model = True)
    #     time1[str(neu_num)].path(tf_path, run_time)
    #     time_model[str(neu_num)].compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['acc', 'mse'])
    #     time_his[str(neu_num)] = time_model[str(neu_num)].fit_generator(train.time_generator, steps_per_epoch = 441, epochs = 200,
    #                                                                   validation_data = val.time_generator, validation_steps = 189,
    #                                                                   callbacks = [time1[str(neu_num)].Tensorboard, time1[str(neu_num)].checkpoint, ES, ES1])
    os.mkdir(os.path.join(tf_path, 'history', run_time))
    for key in dense_list:
        file = open(os.path.join(tf_path, 'history', run_time, 'normal_' + str(key) + '_his.pickle'), 'wb')
        pickle.dump(normal_his[str(key)].history, file)
        file.close
        file = open(os.path.join(tf_path, 'history', run_time, 'shorter_' + str(key) + '_his.pickle'), 'wb')
        pickle.dump(shorter_his[str(key)].history, file)
        file.close
        # file = open(os.path.join(tf_path, 'history', run_time, 'fft_' + str(key) + '_his.pickle'), 'wb')
        # pickle.dump(fft_his[str(key)].history, file)
        # file.close
        # file = open(os.path.join(tf_path, 'history', run_time, 'short_fft_' + str(key) + '_his.pickle'), 'wb')
        # pickle.dump(short_fft_his[str(key)].history, file)
        # file.close
        # file = open(os.path.join(tf_path, 'history', run_time, 'time_' + str(key) + '_his.pickle'), 'wb')
        # pickle.dump(time_his[str(key)].history, file)
        # file.close

#predict
#fft200 = tf.keras.models.load_model()
#predict = fft200.predict(val.fft_data)
#predict70 = predict >0.7  ###get proba > 70%
#true_label = val.label1
#true_mask = np.ma.make_mask(true_label) ### make 1 to True
#np.array_equal(true_mask[0] == predict70[0])### compare true & predict

##read history
## with open(r"path", 'rb') as file:
#   name = pickle.load(file)
# plt.plot(name['acc'])
        
# normal_160 = tf.keras.models.load_model(r"D:\TFcode\Final\Final\checkpoint\0\normal160\normal160.hdf5")
# predict = normal_160.predict(val.data_array_norm)
# predict80 = predict>0.6
# true_label = val.label1
# true_mask = np.ma.make_mask((true_label))
# result = 0
# fail = []
# for i in range(len(predict80)):
#     if not np.array_equal(true_mask[i], predict80[i]):
#         result += 1
#         fail.append(i)
# result = result/len(predict80)*100


# history = {}
# hist_path = r"D:\TFcode\Final\newdropout_model\history"
# for hist_num in os.listdir(hist_path):
#     history[hist_num] = {}
#     subpath = os.path.join(hist_path, hist_num)
#     for hist in os.listdir(subpath):
#         with open(os.path.join(subpath, hist), 'rb') as each_hist:
#             history[hist_num][hist[:-11]] = pickle.load(each_hist)
#         each_hist.close()
        
# history_max = {}
# for num in history:
#     history_max[num] = {}
#     for each in history[num]:
#         history_max[num][each] = max(history[num][each]['acc'])
# # number 0 normal_160 has best performance

# history_eachmax = {}
# for num in history_max:
#     for each in history_max[num]:
#         try:
#             if history_max[num][each] >= history_eachmax[each]:
#                 history_eachmax[each] = history_max[num][each]
#         except:
#             history_eachmax[each] = history_max[num][each]
        
# history_mean = {}
# for num in history_max:
#     for each in history_max[num]:
#         try:
#             history_mean[each] += history_max[num][each]
#         except:
#             history_mean[each] = history_max[num][each]

        
# history_val_max = {}
# for num in history:
#     history_val_max[num] = {}
#     for each in history[num]:
#         history_val_max[num][each] = max(history[num][each]['val_acc'])

# def plot_train_acc(hist):        
#     history = {}
#     hist_path = hist
#     for hist_num in os.listdir(hist_path):
#         history[hist_num] = {}
#         subpath = os.path.join(hist_path, hist_num)
#         for hist in os.listdir(subpath):
#             with open(os.path.join(subpath, hist), 'rb') as each_hist:
#                 try:
#                     history[hist_num][hist[:-11]] = pickle.load(each_hist)
#                 except EOFError:
#                     pass
            
#     history_max = {}
#     for num in history:
#         history_max[num] = {}
#         for each in history[num]:
#             history_max[num][each] = max(history[num][each]['acc'])
#     # number 0 normal_160 has best performance
    
#     history_eachmax = {}
#     for num in history_max:
#         for each in history_max[num]:
#             try:
#                 if history_max[num][each] >= history_eachmax[each]:
#                     history_eachmax[each] = history_max[num][each]
#             except:
#                 history_eachmax[each] = history_max[num][each]
                
#     for each in history_eachmax:
#         history_eachmax[each] = round(history_eachmax[each]*100, 2)
                
#     hist_plot = {}
#     for i in ['time', 'short_fft', 'fft', 'shorter', 'normal']:
#         for num in [40,80,120,160,200,240]:
#             try:
#                 hist_plot[i] = np.append(hist_plot[i], history_eachmax[f"{i}_{num}"])
#             except:
#                 hist_plot[i] = np.array([history_eachmax[f"{i}_{num}"]])
                
#     plt.figure()
#     for inp in hist_plot:
#         plt.plot(np.arange(40,280,40),hist_plot[inp], label=inp, lw = 3)
#         plt.scatter(np.arange(40,280,40), hist_plot[inp])
#     plt.legend(fontsize = 20)
#     plt.xticks(np.arange(40,280,40),fontsize=20)
#     plt.xlabel("Number of Neuron", fontsize = 20)
#     plt.xlim(35,245)
#     plt.yticks(fontsize=20)
#     plt.ylabel('Accuracy (%)', fontsize=20)
#     plt.ylim(68,102)
#     plt.title("Training Accuracy", fontsize=20)
#     plt.tight_layout()
    
#     return history_max, history_eachmax

# def plot_val_acc(hist):        
#     history = {}
#     hist_path = hist
#     for hist_num in os.listdir(hist_path):
#         history[hist_num] = {}
#         subpath = os.path.join(hist_path, hist_num)
#         for hist in os.listdir(subpath):
#             with open(os.path.join(subpath, hist), 'rb') as each_hist:
#                 try:
#                     history[hist_num][hist[:-11]] = pickle.load(each_hist)
#                 except EOFError:
#                     pass
            
#     history_max = {}
#     for num in history:
#         history_max[num] = {}
#         for each in history[num]:
#             history_max[num][each] = max(history[num][each]['val_acc'])
#     # number 0 normal_160 has best performance
    
#     history_eachmax = {}
#     for num in history_max:
#         for each in history_max[num]:
#             try:
#                 if history_max[num][each] >= history_eachmax[each]:
#                     history_eachmax[each] = history_max[num][each]
#             except:
#                 history_eachmax[each] = history_max[num][each]
                
#     for each in history_eachmax:
#         history_eachmax[each] = round(history_eachmax[each]*100, 2)
                
#     hist_plot = {}
#     for i in ['time', 'short_fft', 'fft', 'shorter', 'normal']:
#         for num in [40,80,120,160,200,240]:
#             try:
#                 hist_plot[i] = np.append(hist_plot[i], history_eachmax[f"{i}_{num}"])
#             except:
#                 hist_plot[i] = np.array([history_eachmax[f"{i}_{num}"]])
                
#     plt.figure()
#     for inp in hist_plot:
#         plt.plot(np.arange(40,280,40),hist_plot[inp], label=inp, lw = 3)
#         plt.scatter(np.arange(40,280,40), hist_plot[inp])
#     plt.legend(fontsize = 20)
#     plt.xticks(np.arange(40,280,40),fontsize=20)
#     plt.xlabel("Number of Neuron", fontsize = 20)
#     plt.xlim(35,245)
#     plt.yticks(fontsize=20)
#     plt.ylabel('Accuracy (%)', fontsize=20)
#     plt.ylim(68,102)
#     plt.title("Training Accuracy", fontsize=20)
#     plt.tight_layout()
    
#     return history_max, history_eachmax

## get model's layer
# layer1 = fft_model['160'].layers[1] ##layer0 is input layer
# ly1_weights = layer1.get_weights() ## weights will have 2 list, first is weights, second is bias.
# layer9 = fft_model['160'].layers[9] ##output layer
# ly9_weights = layer9.get_weights()

