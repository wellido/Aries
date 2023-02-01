import numpy as np
import tensorflow as tf
import csv
from utils.data_prepare_tiny import *
import argparse


def cbm(model, x, threshold):
    prediction = model.predict(x)
    max_prob = np.max(prediction, axis=1)
    # print(len(max_prob))
    right_num = len(np.where(max_prob > threshold)[0])
    # print(right_num)
    estimated_acc = right_num
    return estimated_acc


def acc_estimation_run(dataset, data_type, model_type, save_path):
    thresholds = [0.7, 0.8, 0.9]
    if dataset == 'cifar10':

        (x_train, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_train_mean = np.mean(x_train, axis=0)
        if data_type == 'ori':
            x_test = x_test.astype('float32') / 255
            y_test = y_test.reshape(1, -1)[0]
        else:
            x_test = np.load('/scratch/users/qihu/CIFAR-10-C/' + data_type + '.npy')[:10000]
            y_test = np.load('/scratch/users/qihu/CIFAR-10-C/labels.npy')[:10000]

            # x_test = np.load("/home/qhu/qhu-data/Quan_study/datasets/cifar10/CIFAR-10-C/" + data_type + ".npy")[:10000]
            # y_test = np.load("/home/qhu/qhu-data/Quan_study/datasets/cifar10/CIFAR-10-C/labels.npy")[:10000]
            x_test = x_test.astype('float32') / 255
        if model_type == 'resnet20':
            model = tf.keras.models.load_model("models/cifar10/resnet20.h5")
            x_test -= x_train_mean
        elif model_type == 'vgg16':
            model = tf.keras.models.load_model("models/cifar10/VGG16.h5")
        accs = []
        accs.append(data_type)
        for threshold in thresholds:
            estimated_acc = cbm(model, x_test, threshold)
            accs.append(estimated_acc)
        csv_file = open(save_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow(accs)
        finally:
            csv_file.close()
    elif dataset == 'imagenet':
        if data_type == 'ori':
            # hpc
            val_folder = '/scratch/users/qihu/tiny-imagenet-200/val/'
            # dgx
            # val_folder = '/home/qhu/qhu-data/Quan_study/datasets/tiny-imagenet/tiny-imagenet-200/val/'
            test_generator = Tiny_generator_tflite(val_folder)
        else:
            # hpc
            val_folder = '/scratch/users/qihu/Tiny-ImageNet-C/' + data_type + '/1/'
            # dgx
            # val_folder = '/home/qhu/qhu-data/Quan_study/datasets/tiny-imagenet/Tiny-ImageNet-C/' + data_type + '/1/'
            test_generator = Tiny_generator_tflite(val_folder)
        print("run selection")
        if model_type == 'densenet':
            model = tf.keras.models.load_model("models/imagenet/densenet.h5")
        else:
            model = tf.keras.models.load_model("models/imagenet/resnet101.h5")
        print("load data")
        accs = []
        accs.append(data_type)
        for threshold in thresholds:
            estimated_acc = cbm(model, test_generator, threshold)
            accs.append(estimated_acc)
        csv_file = open(save_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow(accs)
        finally:
            csv_file.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-dataset",
                        type=str,
                        default='mnist'
                        )
    parser.add_argument("--data_type", "-data_type",
                        type=str,
                        default='ori'
                        )
    parser.add_argument("--model_type", "-model_type",
                        type=str
                        )
    parser.add_argument("--save_path", "-save_path",
                        type=str,
                        )
    args = parser.parse_args()
    dataset = args.dataset
    data_type = args.data_type
    model_type = args.model_type
    save_path = args.save_path
    acc_estimation_run(dataset, data_type, model_type, save_path)


if __name__ == "__main__":
    main()

# /home/qhu/qhu-data/TS4code/meta_set/
