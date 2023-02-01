import numpy as np
# np.random.seed(698686)
# print("Set Random Seed 698686")
from keras.layers import Convolution2D, MaxPooling2D, Dropout, BatchNormalization
from keras.models import Sequential, load_model
from keras import backend as K
from keras.layers.core import Lambda
from keras.datasets import cifar10
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import optimizers
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
import keras
import tensorflow


def VGG16_clipped(input_shape=None, rate=0.2, nb_classes=10, drop=False):
    # Block 1
    model = Sequential()
    model.add(Convolution2D(64, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block1_conv1', input_shape=input_shape)) #1
    model.add(BatchNormalization(name="batch_normalization_1"))     #2
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(Convolution2D(64, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block1_conv2')) #3
    model.add(BatchNormalization(name="batch_normalization_2"))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))#4

    # Block 2
    model.add(Convolution2D(128, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block2_conv1'))#5
    model.add(BatchNormalization(name="batch_normalization_3"))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(Convolution2D(128, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block2_conv2'))#6
    model.add(BatchNormalization(name="batch_normalization_4"))#7
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))#8

    # Block 3
    model.add(Convolution2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv1'))#9
    model.add(BatchNormalization(name="batch_normalization_5")) #10
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(Convolution2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv2'))#11
    model.add(BatchNormalization(name="batch_normalization_6"))

    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(Convolution2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv3'))#12
    model.add(BatchNormalization(name="batch_normalization_7")) #13
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')) #14
    model.add(Flatten())#15
    model.add(Dense(256, activation='relu', name='dense_1')) #16
    model.add(BatchNormalization(name="batch_normalization_8"))#17
    model.add(Dense(256, activation='relu', name='dense_2'))#18
    model.add(BatchNormalization(name="batch_normalization_9"))#19
    model.add(Dense(nb_classes, activation='softmax', name='dense_3')) #20
    return model


def VGG16_clipped_dropout(input_shape=None, drop_rate=0.5, nb_classes=10, drop=False):
    # Block 1
    inputs = keras.Input(shape=input_shape)
    x = Convolution2D(64, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block1_conv1', input_shape=input_shape)(inputs)
    x = BatchNormalization(name="batch_normalization_1")(x)
    x = Convolution2D(64, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block1_conv2')(x)
    x = BatchNormalization(name="batch_normalization_2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = Convolution2D(128, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block2_conv1')(x)
    x = BatchNormalization(name="batch_normalization_3")(x)
    x = Convolution2D(128, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block2_conv2')(x)
    x = BatchNormalization(name="batch_normalization_4")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv1')(x)
    x = BatchNormalization(name="batch_normalization_5")(x)
    x = Convolution2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv2')(x)
    x = BatchNormalization(name="batch_normalization_6")(x)
    x = Convolution2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv3')(x)
    x = BatchNormalization(name="batch_normalization_7")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', name='dense_1')(x)
    x = Dropout(drop_rate)(x, training=True)
    x = BatchNormalization(name="batch_normalization_8")(x)
    x = Dense(256, activation='relu', name='dense_2')(x)
    x = BatchNormalization(name="batch_normalization_9")(x)
    outputs = Dense(nb_classes, activation='softmax', name='dense_3')(x)
    model = keras.Model(inputs, outputs)
    return model


if __name__ == '__main__':
    nb_class = 10
    num_epoch = 200,
    batchSize = 128
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
    y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

    model = VGG16_clipped(input_shape=x_train.shape[1:], rate=0.4,
                          nb_classes=nb_class)  # VGG16_clipped(input_shape=(32,32,3), rate=0.2, nb_classes=10, drop=False)
    vgg16 = VGG16(weights='imagenet', include_top=False)
    layer_dict = dict([(layer.name, layer) for layer in vgg16.layers])
    for l in model.layers:
        if l.name in layer_dict:
            model.get_layer(name=l.name).set_weights(layer_dict[l.name].get_weights())
    # model.load_weights(weights_path, by_name=True)
    bestmodelname = ""
    checkPoint = ModelCheckpoint(bestmodelname, monitor="val_accuracy", save_best_only=True, verbose=1)
    model.summary()
    num_layers = len(model.layers)
    a = np.arange(num_layers)
    layers = a[(num_layers - 6):-2]
    print(layers)
    lr = 1e-2


    def lr_scheduler(epoch):
        initial_lrate = lr
        drop = 0.9
        epochs_drop = 50.0
        lrate = initial_lrate * np.power(drop,
                                         np.floor((1 + epoch) / epochs_drop))
        return lrate


    reduce_lr = callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    # compile the model with SGD/momentum optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=lr, momentum=0.9),
                  metrics=['accuracy'])
    model.summary()
    # model.save("../new_models/RQ1/VGG16/EGL.h5")
    # csvlog = callbacks.CSVLogger(logpath, separator=',', append=False)

    # data augmentation
    # if you do not want to use data augmentation, comment below codes.
    train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    train_datagen.fit(x_train)
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batchSize)

    # fine-tune the model
    nb_train_samples = x_train.shape[0] // batchSize
    nb_epoch = num_epoch
    his = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples,
        epochs=nb_epoch,
        validation_data=(x_test, y_test),
        callbacks=[checkPoint, reduce_lr])
