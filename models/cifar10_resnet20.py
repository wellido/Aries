from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from keras.regularizers import l2
from keras.layers import Conv2D, Dense, Input, add, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dropout
from keras.layers import BatchNormalization
from keras import optimizers, regularizers
from keras.models import Sequential, Model, load_model

from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten,MaxPool2D, Dropout

from keras.models import Model
from keras.datasets import cifar10
import numpy as np
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
import tensorflow as tf
import argparse


def resnet20_drop(input_shape, num_classes, drop_rate=0.5):
    input_tensor = Input(shape=input_shape)
    weight_decay = 1e-6
    stack_n = 3
    def residual_block(intput, out_channel, drop=False, increase=False):
        if increase:
            stride = (2, 2)
        else:
            stride = (1, 1)

        pre_bn = BatchNormalization()(intput)
        pre_relu = Activation('relu')(pre_bn)

        conv_1 = Conv2D(out_channel, kernel_size=(3, 3), strides=stride, padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(pre_relu)
        bn_1 = BatchNormalization()(conv_1)
        relu1 = Activation('relu')(bn_1)
        conv_2 = Conv2D(out_channel, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(relu1)
        if drop:
            conv_2 = Dropout(drop_rate)(conv_2, training=True)
        if increase:
            projection = Conv2D(out_channel,
                                kernel_size=(1, 1),
                                strides=(2, 2),
                                padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(intput)
            block = add([conv_2, projection])
        else:
            block = add([intput, conv_2])
        return block

        # build model
        # total layers = stack_n * 3 * 2 + 2
        # stack_n = 5 by default, total layers = 32, which is resnet32
        # input: 32x32x3 output: 32x32x16

    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(input_tensor)
    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x, 16, increase=False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, 32, increase=True)
    for _ in range(1, stack_n):
        x = residual_block(x, 32, increase=False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, 64, increase=True)
    for _ in range(1, stack_n):
        x = residual_block(x, 64, drop=True, increase=False)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(num_classes, name='before_softmax',
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('softmax')(x)
    model = Model(input_tensor, x, name='res20-none')
    return model


if __name__ == '__main__':
    input_shape = (32, 32, 3)
    model = resnet20_drop(input_shape, 10)
    model.summary()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    model_filename = ""

    checkpoint = ModelCheckpoint(model_filename,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 period=1)
    # cbks = [checkpoint]
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=200,
                        validation_data=(x_test, y_test),
                        verbose=1,
                        callbacks=[checkpoint]
                        )

