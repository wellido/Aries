from Aries import *
from tensorflow_addons.optimizers import SGDW
tf.keras.optimizers.SGDW = SGDW

(x_train, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_train_mean = np.mean(x_train, axis=0)
x_test = x_test.astype('float32') / 255
y_test = y_test.reshape(1, -1)[0]
ori_model = tf.keras.models.load_model("models/cifar10/resnet20.h5")
drop_model = tf.keras.models.load_model("models/cifar10/resnet20_drop.h5")
x_test -= x_train_mean
base_map = drop_check_func(ori_model, drop_model, x_test[:5000], y_test[:5000], 50)
base_acc = 0.8728 # accuracy of labeled test data
estimated_acc, acc1, acc2 = Aries_estimation(x_test[5000:], drop_model=drop_model, base_map=base_map, base_acc=base_acc, section_num=50)
print("Acc: {}, {}, {}".format(estimated_acc, acc1, acc2))

