import os
import numpy as np
from struct import unpack
import gzip

# 读取 MNIST 数据集中的图像数据
def __read_image(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28, 28, 1)
    return img

# 读取 MNIST 数据集中的标签数据
def __read_label(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.frombuffer(f.read(), dtype=np.uint8)
    return lab

# 对图像进行归一化处理
def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img

# 将标签进行 one-hot 编码
def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab

# 加载 MNIST 数据集
def load_mnist(path, normalize=True, one_hot=True):
    x_train_path = os.path.join(path, 'train-images-idx3-ubyte.gz')
    y_train_path = os.path.join(path, 'train-labels-idx1-ubyte.gz')
    x_test_path = os.path.join(path, 't10k-images-idx3-ubyte.gz')
    y_test_path = os.path.join(path, 't10k-labels-idx1-ubyte.gz')

    image = {
        'train': __read_image(x_train_path),
        'test': __read_image(x_test_path)
    }

    label = {
        'train': __read_label(y_train_path),
        'test': __read_label(y_test_path)
    }

    # 对图像进行归一化处理
    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    # 对标签进行 one-hot 编码
    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])

    return (image['train'], label['train']), (image['test'], label['test'])

# 获取 MNIST 数据集
def get_mnist(data_dir='./Mnist'):
    (x_train, y_train), (x_test, y_test) = load_mnist(data_dir)
    return (x_train, y_train), (x_test, y_test)
