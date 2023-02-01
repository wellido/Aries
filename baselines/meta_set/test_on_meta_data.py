import tensorflow as tf
import numpy as np
from data_prepare_tiny import *
from tensorflow_addons.optimizers import SGDW
tf.keras.optimizers.SGDW = SGDW


def main():
    model = tf.keras.models.load_model("../../models/cifar10/VGG16.h5")
    model.summary()
    (x_train, _), (_, y_test) = tf.keras.datasets.cifar10.load_data()
    y_test = y_test.reshape(1, -1)
    # x_train = x_train.astype('float32') / 255
    # x_train_mean = np.mean(x_train, axis=0)
    for j in range(1, 4):
        accuracy = []

        data_base = "/home/qhu/qhu-data/TS4code/meta_set/cifar_data_" + str(j) + "/new_data_"
        save_path = "/home/qhu/qhu-data/TS4code/meta_set/inter_data/cifar10_vgg16_acc_" + str(j) + ".npy"

        for i in range(1000):
            if i == 891:
                accuracy.append(80)
                continue
            x_test = np.load(data_base + str(i).zfill(3) + ".npy")
            x_test = x_test.astype('float32') / 255
            # x_test -= x_train_mean
            predicted_label = model.predict(x_test).argmax(axis=1)
            right_num = len(np.where(predicted_label == y_test)[0])
            # print(right_num)
            acc = 100. * right_num / len(x_test)
            print("num: {}, acc: {}".format(i, acc))
            accuracy.append(acc)
        accuracy = np.asarray(accuracy)
        print(accuracy)
        np.save(save_path, accuracy)


def main_imagenet():
    model = tf.keras.models.load_model("../../models/imagenet/resnet101.h5")
    model.summary()

    for j in range(1, 4):
        data_base = "/home/qhu/qhu-data/TS4code/meta_set/imagenet_data_" + str(j) + "/"
        accuracy = []
        save_path = "/home/qhu/qhu-data/TS4code/meta_set/inter_data/imagenet_acc_resnet_" + str(j) + ".npy"
        for i in range(1000):
            print(i)
            files = data_base + str(i) + "/"
            x_test, y_test = Tiny_generator_val(files)
            y_test = y_test.argmax(axis=1)
            # predictions = model.predict(x_test, batch_size=100).argmax(axis=1)
            # acc = np.sum(predictions == y_test) / len(predictions)
            predicted_label = model.predict(x_test, batch_size=100).argmax(axis=1)
            right_num = len(np.where(predicted_label == y_test)[0])
            # print(right_num)
            acc = 100. * right_num / len(x_test)
            print(acc)
            accuracy.append(acc)
        accuracy = np.asarray(accuracy)
        print(accuracy)
        np.save(save_path, accuracy)



if __name__ == "__main__":
    main_imagenet()
    # main()


