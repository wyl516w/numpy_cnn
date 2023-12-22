import numpy as np
import os
import pickle

# 读取 CIFAR-10 数据集中的 pickle 文件
def __unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 读取 CIFAR-10 数据集中的一个 batch 文件
def __read_data_batch(file):
    d = __unpickle(file)
    data = d[b'data']
    labels = d[b'labels']
    return data, labels

# 对图像进行归一化处理
def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img.reshape(-1, 3, 32, 32)  # CIFAR-10 图像大小为 32x32x3

# 将标签进行 one-hot 编码
def __one_hot_label(label):
    lab = np.zeros((len(label), 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab

# 加载 CIFAR-10 数据集
def load_cifar10(data_dir, normalize=True, one_hot=True):
    image_train, label_train = [], []

    # 读取训练集数据
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')  
        data, labels = __read_data_batch(batch_file)
        image_train.append(data)
        label_train.extend(labels)

    # 将训练集数据拼接为一个大的数组
    image_train = np.concatenate(image_train, axis=0)
    label_train = np.array(label_train)

    # 读取测试集数据
    test_file = os.path.join(data_dir, 'test_batch')
    image_test, label_test = __read_data_batch(test_file)

    # 对图像进行归一化处理
    if normalize:
        image_train = __normalize_image(image_train)
        image_test = __normalize_image(image_test)

    # 对标签进行 one-hot 编码
    if one_hot:
        label_train = __one_hot_label(label_train)
        label_test = __one_hot_label(label_test)

    return (image_train, label_train), (image_test, label_test)

# 获取 CIFAR-10 数据集
def get_cifar10(data_dir = './CIFAR-10'):
    (x_train, y_train), (x_test, y_test) = load_cifar10(data_dir)
    return (x_train, y_train), (x_test, y_test)
