from tensorflow.keras.preprocessing.image import ImageDataGenerator
from art.data_generators import KerasDataGenerator


class imagenet_config:
    def set_defaults(self):
        self.batch_size = 64
        self.epochs = 30


def preprocess(training_folder, test_folder):
    """
    Returns a tuple of the form (training generator, validation generator).
    Image preprocessing is handled here via Keras ImageDataGenerator
    """
    print("Reading images from: %s", training_folder)

    # Create object to read training images w/ preprocessing
    #
    # Standardize each image to zero mean unit variance with
    #   with random brightness perturbartions and horizontal flips
    #
    # Training / validation split is specified here
    # TODO get all these params from FLAGS
    config = imagenet_config()
    config.set_defaults()
    # train_datagen = ImageDataGenerator(
    #         samplewise_center=True,
    #         samplewise_std_normalization=True,
    #         rescale=1./255,
    #         horizontal_flip=True,
    #         data_format='channels_last',
    #         # validation_split=0.5,
    # )

    train_datagen = ImageDataGenerator(
        # featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=True,  # don't set each sample mean to 0
        # featurewise_std_normalization=True,  # divide all inputs by std of the dataset
        samplewise_std_normalization=True,  # don't divide each input by its std
        zca_whitening=False,  # don't apply ZCA whitening.
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180).
        horizontal_flip=True,  # randomly flip horizontal images.
        vertical_flip=False,  # don't randomly flip vertical images.
        zoom_range=0.1,  # slightly zoom in.
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1. / 255,
        data_format='channels_last',
    )

    # Create generator to yield a training set from directory
    train_generator = train_datagen.flow_from_directory(
            training_folder,
            # subset='training',
            target_size=(64, 64),
            class_mode='sparse',
            batch_size=config.batch_size,
            seed=42
    )

    test_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rescale=1. / 255,
        horizontal_flip=True,
        data_format='channels_last',
        # validation_split=0.5,
    )

    # Create generator to yield a validation set from directory
    val_generator = test_datagen.flow_from_directory(
            test_folder,
            # subset='validation',
            target_size=(64, 64),
            class_mode='sparse',
            batch_size=config.batch_size,
            seed=42
    )

    return train_generator, val_generator


# Preprocess image to fit in Xception Model.
class CustomImageDataGen(ImageDataGenerator):  # Inheriting class ImageDataGenerator and manually standardize each input image (x)
    def standardize(self, x):
        if self.featurewise_center:
            x /= 255.
            x -= [0.485, 0.456, 0.406]
            x /= [0.229, 0.224, 0.225]
        return x


def Tiny_generator_tflite(test_folder):
    data_generator = CustomImageDataGen(
        horizontal_flip=True,
        # featurewise_center=True
    )
    #   Loading Validation Data...
    test_data_generator = data_generator.flow_from_directory(test_folder,
                                                             target_size=(224, 224),
                                                             batch_size=50,
                                                             seed=1,
                                                             shuffle=False)
    return test_data_generator


def Tiny_generator_val(test_folder):
    data_generator = CustomImageDataGen(
        horizontal_flip=True,
        # featurewise_center=True
    )
    test_data_generator = data_generator.flow_from_directory(test_folder,
                                                             target_size=(224, 224),
                                                             batch_size=10000,
                                                             seed=1,
                                                             shuffle=False)
    x_test = None
    y_test = None
    for x, y in test_data_generator:
        x_test = x
        y_test = y
        break
    return x_test, y_test


if __name__ == '__main__':
    train_path = '/Users/qiang.hu/PycharmProjects/tint-imagenet/tiny-imagenet-200/train/'
    test_path = '/Users/qiang.hu/PycharmProjects/tiny-imagenet/tiny-imagenet-200/val/'
    # test_path = '/home/qhu/qhu-data/Quan_study/datasets/tiny-imagenet/tiny-imagenet-200/val/'
    x_test, y_test = Tiny_generator_val(test_path)
    print(x_test.shape)
    # train_generator, val_generator = preprocess(train_path, test_path)
    # train_generator = img_process(train_path)
    # val_generator = img_process(test_path)
    # n = 1
    # for x, y in val_generator:
    #     # print(x)
    #     print(y[0])
    #     n += 1
    #     if n == 10:
    #         break
    # print(dir(val_generator))
    # data_gen = KerasDataGenerator(
    #     val_generator,
    #     size=10000,
    #     batch_size=64
    # )
    #
    # x, y = data_gen.get_batch()
    # print(x.shape)
    # print(y.shape)
