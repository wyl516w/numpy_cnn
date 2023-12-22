from sad_layer import FullConnectedLayer,Conv2DLayer,ReLuLayer,SoftmaxCrossEntropyLossLayer,MaxPooling2DLayer
import numpy as np

sample_layer_list=[
    FullConnectedLayer(784,256),
    ReLuLayer(),
    FullConnectedLayer(256,64),
    ReLuLayer(),
    FullConnectedLayer(64,10),
]
sample_loss=SoftmaxCrossEntropyLossLayer()

class NeuralNetwork():
    def __init__(self,layers=sample_layer_list,lossfunction=sample_loss) -> None:
        self.layers=layers
        self.lossfunction=lossfunction
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
    def forward(self,input):
        for layer in self.layers:
            input=layer.forward(input)
        return self.lossfunction.forward(input)
    def loss(self,y):
        return self.lossfunction.loss(y)
    def backward(self):
        grad=self.lossfunction.backward()
        for layer in reversed(self.layers):
            grad=layer.backward(grad)
    def update(self,learning_rate=0.01):
        for layer in self.layers:
            layer.update()
    def predcit(self,x_test):
        return np.argmax(self.forward(x_test),axis=1)
    def accuracy(self,x_test,y_test):
        onehotsize=y_test.shape[1]
        y_predict=self.forward(x_test)
        y_predict=np.argmax(y_predict,axis=1)
        y_test=np.argmax(y_test,axis=1)
        return np.mean(y_predict==y_test),np.array([np.mean(y_predict[y_test==idx]==idx) for idx in range(onehotsize)]),np.array([[np.mean(y_predict[y_test==idx]==jdx) for jdx in range(onehotsize)] for idx in range(onehotsize)])
    def testshape(self,input):
        for layer in self.layers:
            input=layer.forward(input)
            print(input.shape)
        return self.lossfunction.forward(input)
    def init_para(self):
        for layer in self.layers:
            layer.init()
    def save_para(self,filedir):
        save_paras=[]
        for layer in self.layers:
            save_paras.append(layer.save())
        np.save(filedir,save_paras)    
    def load_para(self,filedir):
        try:
            load_para=np.load(filedir,allow_pickle=True)
            for layer,para in zip(self.layers,load_para):
                layer.load(para)
        except:
            print("init para")
            self.init_para()
