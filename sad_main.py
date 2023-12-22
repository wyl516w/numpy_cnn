import numpy as np
from sad_construct import *
from sad_read_cifar10 import get_cifar10
(x_train,y_train),(x_test,y_test)=get_cifar10()
x_train=x_train.reshape(-1,3*32*32)
x_test=x_test.reshape(-1,3*32*32)
cifar10_train_data = np.append(x_train, y_train, axis=1)
cifar10_test_data = np.append(x_test, y_test, axis=1)
cifar10_shape=(3,32,32)

train_and_test(
    mynetwork=cnn_cifar10,
    train_data=cifar10_train_data,
    test_data=cifar10_test_data,
    data_shape=cifar10_shape,
    onehotsize=10,
    batch_size=16,
    epoch_num=50,
    half_learning_rate_time=5,
    pth="model/cnn_cifar10.npy"
)
