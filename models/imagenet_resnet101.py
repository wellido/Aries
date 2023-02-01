import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Activation, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow_addons.optimizers import SGDW
tf.keras.optimizers.SGDW = SGDW
import tensorflow_addons as tfa
from data_prepare_tiny import *

def schedule(epoch_idx):
    if (epoch_idx + 1) < 10:
        return 1e-03
    elif (epoch_idx + 1) < 20:
        return 1e-04  # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < 30:
        return 1e-05
    return 1e-05

def ResNet101_model():
    base_model = tf.keras.applications.ResNet101V2(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=200
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(200)(x)
    predictions = Activation('softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def ResNet101_drop(drop_rate=0.5):
    base_model = tf.keras.applications.ResNet101V2(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=200
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(drop_rate)(x, training=True)
    x = Dense(200)(x)
    predictions = Activation('softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


if __name__ == '__main__':
    save_path = ""
    model = ResNet101_drop()
    config = imagenet_config()
    config.set_defaults()
    # model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    training_folder = '../datasets/tiny-imagenet/tiny-imagenet-200/train/'
    val_folder = '../datasets/tiny-imagenet/tiny-imagenet-200/val/'
    train_generator, val_generator = Tiny_generator(training_folder, val_folder)
    checkPoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy",
                                                    save_best_only=True, verbose=0)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(schedule=schedule)
    cbs = [lr_schedule, checkPoint]
    model.compile(loss='categorical_crossentropy',
                  optimizer=tfa.optimizers.SGDW(lr=1e-03, momentum=0.9, weight_decay=1e-4),
                  # optimizer=optimizers.Adam(lr=1e-03),
                  metrics=['accuracy'])
    model.fit_generator(generator=train_generator,
                        epochs=config.epochs,
                        steps_per_epoch=100000 // config.batch_size,
                        validation_data=val_generator,
                        validation_steps=10000 // config.batch_size,
                        use_multiprocessing=True,
                        callbacks=cbs)


