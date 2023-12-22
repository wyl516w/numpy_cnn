from sad_layer import *
from sad_network import NeuralNetwork as Network
def train_and_test(mynetwork:Network,train_data,test_data,data_shape,onehotsize=10,batch_size=32,epoch_num=20,learning_rate=0.01,half_learning_rate_time=2,pth=None):
    if pth!=None:
        mynetwork.load_para(pth)
    else:
        mynetwork.init_para()
    #mynetwork.testshape(test_data[1:2,:-onehotsize].reshape(-1,*data_shape))
    for epoch_id in range(1,epoch_num+1):
        loss=0
        np.random.shuffle(train_data)
        print("epoch",epoch_id,end=":\n")
        batch_num=train_data.shape[0]//batch_size
        for idx_batch in range(batch_num):
            batch_x = train_data[idx_batch*batch_size:(idx_batch+1)*batch_size,:-onehotsize].reshape(-1,*data_shape)
            batch_y = train_data[idx_batch*batch_size:(idx_batch+1)*batch_size,-onehotsize:]
            mynetwork.zero_grad()
            mynetwork.forward(batch_x)
            l=mynetwork.loss(batch_y)
            loss+=l
            mynetwork.backward()
            mynetwork.update(learning_rate)
            if epoch_id%half_learning_rate_time==0: learning_rate/=2
            print("\r loss:%.9f"%l,"\tprocess:", int(idx_batch/(train_data.shape[0]//batch_size)*10000)/100,end="%\t   ")
        x_test = test_data[:,:-onehotsize].reshape(-1,*data_shape)
        y_test = test_data[:,-onehotsize:]
        accuracy,accuracy_list,accuracy_crosslist =mynetwork.accuracy(x_test,y_test)
        print("\r Loss:%.9f"%(loss/batch_num),"\tAccuracy:",accuracy,[round(i,5) for i in accuracy_list])
        if pth!=None:
            mynetwork.save_para(pth)

def only_test(mynetwork:Network,test_data,data_shape,onehotsize=10,pth=None):
    if pth!=None:
        mynetwork.load_para(pth)
    else:
        mynetwork.init_para()
    x_test = test_data[:,:-onehotsize].reshape(-1,*data_shape)
    y_test = test_data[:,-onehotsize:]
    accuracy,accuracy_list,accuracy_crosslist=mynetwork.accuracy(x_test,y_test)
    print("Accuracy:",accuracy,accuracy_list)
    print("Cross Accuracy:")
    print(accuracy_crosslist)
    

LeNet5_cifar10=Network(
    layers=[
        Conv2DLayer(3,6,5,0),
        ReLuLayer(),
        MaxPooling2DLayer(2,2),
        Conv2DLayer(6,16,5,0),
        ReLuLayer(),
        MaxPooling2DLayer(2,2),
        FullConnectedLayer(16*5*5,120),
        ReLuLayer(),
        FullConnectedLayer(120,84),
        ReLuLayer(),
        Flatten(),
        FullConnectedLayer(84,10)
    ],
    lossfunction=SoftmaxCrossEntropyLossLayer()
)

cnn_cifar10=Network(
    layers=[
        Conv2DLayer(3,16,3,1),
        ReLuLayer(),
        MaxPooling2DLayer(2,2),
        Conv2DLayer(16,32,3,1),
        ReLuLayer(),
        MaxPooling2DLayer(2,2),
        Conv2DLayer(32,64,3,1),
        ReLuLayer(),
        MaxPooling2DLayer(2,2),
        Conv2DLayer(64,32,3,1),
        ReLuLayer(),
        MaxPooling2DLayer(2,2),
        Flatten(),
        FullConnectedLayer(32*4,32),
        ReLuLayer(),
        FullConnectedLayer(32,10)
    ],
    lossfunction=SoftmaxCrossEntropyLossLayer()
)

Fc_cifar10=Network(
    layers=[
        FullConnectedLayer(3*32*32,1024),
        ReLuLayer(),
        FullConnectedLayer(1024,256),
        ReLuLayer(),
        FullConnectedLayer(256,64),
        ReLuLayer(),
        FullConnectedLayer(64,10)
    ],
    lossfunction=SoftmaxCrossEntropyLossLayer()
)

fc_mnist=Network(
    layers=[
        FullConnectedLayer(784,256),
        ReLuLayer(),
        FullConnectedLayer(256,64),
        ReLuLayer(),
        FullConnectedLayer(64,10)
    ],
    lossfunction=SoftmaxCrossEntropyLossLayer()
)