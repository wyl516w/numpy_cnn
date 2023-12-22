import numpy as np
from sad_read_mnist import get_mnist
(x_train,y_train),(x_test,y_test)=get_mnist()
x_train=x_train.reshape(-1,1*28*28)
# from sad_read_cifar10 import get_cifar10
# (x_train,y_train),(x_test,y_test)=get_cifar10()
# print(x_train.shape,x_test.shape)
# x_train=x_train.reshape(-1,3*32*32)


from sad_network import NeuralNetwork as Network
train_data = np.append(x_train, y_train, axis=1)

from sad_layer import *
batch_size=32

sample_layer_list=[
    AveragePooling2DLayer(2,2),
    FullConnectedLayer(14*14,64),
    ReLuLayer(),
    FullConnectedLayer(64,32),
    ReLuLayer(),
    FullConnectedLayer(32,10)
]
cov1=[#3*32*32
    Conv2DLayer(1,16,3,0),
    ReLuLayer(),
    AveragePooling2DLayer(2,2),
    Conv2DLayer(16,32,3,0),
    ReLuLayer(),
    AveragePooling2DLayer(2,2),
    Conv2DLayer(32,64,3,0),
    ReLuLayer(),
    MaxPooling2DLayer(2,2),
    FullConnectedLayer(64,30),
    FullConnectedLayer(30,10),
]


mynetwork=Network(layers=sample_layer_list)
def ac(network,x_test,y_test):
    y_predict=network.forward(x_test.reshape(-1,1,28,28))
    total=y_test.shape[0]
    y_predict=np.argmax(y_predict,axis=1)
    y_test=np.argmax(y_test,axis=1)
    return float(np.mean(y_predict==y_test))

for i in range(200):
    loss=0
    np.random.shuffle(train_data)
    print("epoch",i,end=":\n")
    for idx_batch in range(train_data.shape[0]//batch_size):
        batch_x = train_data[idx_batch*batch_size:(idx_batch+1)*batch_size,:-10].reshape(-1,1,28,28)
        batch_y = train_data[idx_batch*batch_size:(idx_batch+1)*batch_size,-10:]
        #mynetwork.test(batch_x)
        mynetwork.zero_grad()
        mynetwork.forward(batch_x)
        l=mynetwork.loss(batch_y)
        loss+=l
        mynetwork.backward()
        mynetwork.update(0.1)
        print("\r loss:%.9f"%l,"\tprocess:", int(idx_batch/(train_data.shape[0]//batch_size)*10000)/100,end="%\t   ")
    print("\r Loss:%.9f"%loss,"\tAccuracy:",ac(mynetwork,x_test,y_test),"\t\t")
    
    # with open("history.txt","w+") as f:
    #     f.writelines(["Epoch%d"%i,"Loss:%.9f"%loss,"\tAccuracy:%.4f"%ac(mynetwork,x_test,y_test),"\n"])
