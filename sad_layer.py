import numpy as np

# 计算一个形状的大小
def size(x) -> int:
    if(isinstance(x, int)): return x
    ret=1
    for i in x:
        ret*=i
    return ret

# 将输入数据转换为列矩阵
def im2col(input_data, filter_h:int, filter_w:int, stride = 1, pad=0):
    N, C, H, W = input_data.shape
    stride_h,stride_w=(stride,stride) if isinstance(stride,int) else stride
    out_h = (H + 2*pad - filter_h)//stride_h + 1
    out_w = (W + 2*pad - filter_w)//stride_w + 1
    img = np.pad(input_data, [(0,0),(0,0),(pad,pad),(pad,pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    # 提取卷积核大小的元素
    for y in range(filter_h):
        y_max = y + stride_h*out_h
        for x in range(filter_w):
            x_max = x + stride_w*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride_h, x:x_max:stride_w]
    # 塑造为列矩阵
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

# 将列矩阵转换回原始输入数据形状
def col2im(col, input_shape, filter_h:int, filter_w:int, stride=1, pad=0):
    stride_h,stride_w=(stride,stride) if isinstance(stride,int) else stride
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride_h + 1
    out_w = (W + 2*pad - filter_w)//stride_w + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2*pad + stride_h - 1, W + 2*pad + stride_w - 1))
    for y in range(filter_h):
        y_max = y + stride_w*out_h
        for x in range(filter_w):
            x_max = x + stride_w*out_w
            img[:, :, y:y_max:stride_w, x:x_max:stride_h] += col[:, :, y, x, :, :]
    # 移除填充以获得原始输入数据形状
    return img[:, :, pad:H + pad, pad:W + pad]

# 一个网络层（啥也不干）
class Layer(): 
    # 初始化网络层
    def __init__(self):
        pass
    # 重新初始化参数
    def init(self):
        pass
    # 清除累计梯度
    def zero_grad(self):
        pass
    # 前向传播
    def forward(self,input_data):
        self.input_data=input_data
        return input_data
    # 反向传播
    def backward(self,out_grad):
        return out_grad.reshspae(self.input_data.shape)
    # 更新参数
    def update(self,learning_rate=0.01):
        pass
    def save(self):
        return {}
    def load(self,dict):
        pass

#####连接层
class FullConnectedLayer(Layer):
    """
    全连接层
    """
    def __init__(self,input_shape,output_shape,std=None):
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.input_size=size(self.input_shape)
        self.output_size=size(self.output_shape)
        if std==None: std=(2/(self.input_size+self.output_size))**0.5
        self.std=std
        self.w=np.random.normal(loc=0.0, scale=std, size=(self.input_size,self.output_size))
        self.b=np.zeros([1,self.output_size])
        self.grad_w=np.zeros_like(self.w)
        self.grad_b=np.zeros_like(self.b)
    def init(self):
        self.w=np.random.normal(loc=0.0, scale=self.std, size=(self.input_size,self.output_size))
        self.b=np.zeros([1,self.output_size])
    def zero_grad(self):
        self.grad_w=np.zeros_like(self.grad_w)
        self.grad_b=np.zeros_like(self.grad_b)
    def forward(self,input_data):
        self.input_data=input_data.reshape(-1,self.input_size)
        self.output_data=np.matmul(self.input_data,self.w)+self.b
        return self.output_data
    def backward(self,out_grad):
        out_grad=out_grad.reshape(-1,self.output_size)
        self.grad_w+=np.dot(self.input_data.T,out_grad)
        self.grad_b+=np.sum(out_grad, axis=0)
        grad=np.dot(out_grad, self.w.T)
        return grad
    def update(self,learning_rate=0.01):
        self.w=self.w-learning_rate*self.grad_w
        self.b=self.b-learning_rate*self.grad_b
    def save(self):
        return {'w':self.w,'b':self.b}
    def load(self, dict):
        self.w = dict['w']
        self.b = dict['b']
        
class Conv2DLayer(Layer):
    """
    卷积层
    利用col2im和im2col优化
    """
    def __init__(self, input_channels=1, output_channels=1, kernel_size=3, padding=0, stride=1, std=None):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.padding = padding
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        if std==None: std=(2/(size(self.kernel_size)*(self.input_channels+output_channels)))**0.5
        self.std=std
        self.w=np.random.normal(0.0,std,(self.output_channels,self.input_channels,*self.kernel_size))
        self.b=np.zeros(self.output_channels)
        self.grad_w=np.zeros_like(self.w)
        self.grad_b=np.zeros_like(self.b)
    def init(self):
        self.w=np.random.normal(0.0,self.std,(self.output_channels,self.input_channels,*self.kernel_size))
        self.b=np.zeros(self.output_channels)
    def zero_grad(self):
        self.grad_w=np.zeros_like(self.w)
        self.grad_b=np.zeros_like(self.b)
    def forward(self,input_data):
        self.input_data=input_data
        batch_size,_,h,w=self.input_data.shape
        out_h=(h+2*self.padding-self.kernel_size[0])//self.stride[0]+1
        out_w=(w+2*self.padding-self.kernel_size[1])//self.stride[1]+1
        self.col=im2col(input_data,*self.kernel_size,self.stride,self.padding)
        self.output_data=(np.dot(self.col,self.w.reshape(self.output_channels,-1).T)+self.b).reshape(batch_size,out_h,out_w,-1).transpose(0,3,1,2)
        return self.output_data
    def backward(self, out_grad):
        grad=out_grad.reshape(self.output_data.shape)
        grad=grad.transpose(0,2,3,1).reshape(-1,self.output_channels)
        self.grad_b+=np.sum(grad,axis=0)
        self.grad_w+=np.dot(self.col.T,grad).transpose(1,0).reshape(self.grad_w.shape)
        dcol=np.dot(grad,im2col(self.w,*self.kernel_size,1,0))    
        return col2im(dcol,self.input_data.shape,*self.kernel_size,self.stride,self.padding)
    def update(self,learning_rate=0.01):
        self.w=self.w-learning_rate*self.grad_w
        self.b=self.b-learning_rate*self.grad_b
    def save(self):
        return {'w':self.w,'b':self.b}
    def load(self, dict):
        self.w = dict['w']
        self.b = dict['b']
        
#####调整层
class Flatten(Layer):
    def forward(self,input_data):
        self.input_data=input_data
        return self.input_data.reshape(input_data.shape[0],-1)
    def backward(self,out_grad):
        return out_grad.reshape(self.input_data.shape)



#####池化层

class MaxPooling2DLayer(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.input_data = None
        self.mask = None
    def forward(self, input_data):
        self.input_data = input_data
        N, C, H, W = input_data.shape
        out_h = int(1 + (H - self.pool_size[0]) / self.stride[0])
        out_w = int(1 + (W - self.pool_size[1]) / self.stride[1])
        col = im2col(input_data,*self.pool_size, self.stride,0)
        col = col.reshape(-1,self.pool_size[0]*self.pool_size[1])
        self.arg_max = np.argmax(col, axis=1)
        self.output_data = np.max(col,axis=1).reshape(N, out_h, out_w, C).transpose(0,3,1,2)
        return self.output_data
    def backward(self, out_grad):
        out_grad = out_grad.reshape(self.output_data.shape).transpose(0, 2, 3, 1)
        pool_size = size(self.pool_size)
        dmax = np.zeros((out_grad.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = out_grad.flatten()
        dmax = dmax.reshape(out_grad.shape + (pool_size, ))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        grad = col2im(dcol, self.input_data.shape, *self.pool_size, self.stride, 0)
        return grad 
    
class MinPooling2DLayer(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.input_data = None
        self.mask = None
    def forward(self, input_data):
        self.input_data = input_data
        N, C, H, W = input_data.shape
        out_h = int(1 + (H - self.pool_size[0]) / self.stride[0])
        out_w = int(1 + (W - self.pool_size[1]) / self.stride[1])
        col = im2col(input_data,*self.pool_size, self.stride,0)
        col = col.reshape(-1,self.pool_size[0]*self.pool_size[1])
        self.arg_min = np.argmin(col, axis=1)
        self.output_data = np.min(col,axis=1).reshape(N, out_h, out_w, C).transpose(0,3,1,2)
        return self.output_data
    def backward(self, out_grad):
        out_grad = out_grad.reshape(self.output_data.shape).transpose(0, 2, 3, 1)
        pool_size = size(self.pool_size)
        dmin = np.zeros((out_grad.size, pool_size))
        dmin[np.arange(self.arg_min.size), self.arg_min.flatten()] = out_grad.flatten()
        dmin = dmin.reshape(out_grad.shape + (pool_size, ))
        dcol = dmin.reshape(dmin.shape[0] * dmin.shape[1] * dmin.shape[2], -1)
        grad = col2im(dcol, self.input_data.shape, *self.pool_size, self.stride, 0)
        return grad     


class AveragePooling2DLayer(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.input_data = None
        self.mask = None
    def forward(self, input_data):
        self.input_data = input_data
        N, C, H, W = input_data.shape
        out_h = int(1 + (H - self.pool_size[0]) / self.stride[0])
        out_w = int(1 + (W - self.pool_size[1]) / self.stride[1])
        col = im2col(input_data,*self.pool_size, self.stride,0)
        col = col.reshape(-1,self.pool_size[0]*self.pool_size[1])
        self.asize = col.shape[0]
        self.output_data = np.mean(col,axis=1).reshape(N, out_h, out_w, C).transpose(0,3,1,2)
        return self.output_data
    def backward(self, out_grad):
        out_grad = out_grad.reshape(self.output_data.shape).transpose(0, 2, 3, 1)
        pool_size = size(self.pool_size)
        dave = np.zeros((out_grad.size, pool_size))
        dave += (out_grad.flatten()/(pool_size)).reshape(out_grad.size,1)
        dave = dave.reshape(out_grad.shape + (pool_size, ))
        dcol = dave.reshape(dave.shape[0] * dave.shape[1] * dave.shape[2], -1)
        grad = col2im(dcol, self.input_data.shape, *self.pool_size, self.stride, 0)
        return grad     
        
    
          
#####激活层

class ReLuLayer(Layer):
    def forward(self,input_data):
        self.input_data=input_data
        self.output_data=np.maximum(0,input_data)
        return self.output_data
    def backward(self,out_grad):
        grad=out_grad.reshape(self.input_data.shape)
        grad[self.input_data<0]=0
        return grad

class SigmoidLayer(Layer):
    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = 1 / (1 + np.exp(-input_data))
        return self.output_data

    def backward(self, out_grad):
        out_grad=out_grad.reshape(self.input_data.shape)
        sigmoid_grad = self.input_data * (1 - self.input_data)
        grad = out_grad * sigmoid_grad 
        return grad

class TanhLayer(Layer):
    def forward(self, input_data):
        self.input_data = input_data
        output_data = np.tanh(input_data)
        return output_data
    def backward(self, out_grad):
        out_grad.reshape(self.input_data.shape)
        tanh_grad = 1 - np.tanh(self.input_data)**2
        grad = out_grad * tanh_grad
        return grad

class LeakyReLULayer(Layer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = np.maximum(self.alpha * input_data, input_data)
        return self.output_data
    def backward(self, out_grad):
        out_grad.reshape(self.input_data.shape)
        grad = out_grad.copy()
        grad[self.input_data < 0] *= self.alpha
        return grad
    
class ELULayer(Layer):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = np.where(input_data > 0, input_data, self.alpha * (np.exp(input_data) - 1))
        return self.output_data
    def backward(self, out_grad):
        out_grad.reshape(self.input_data.shape)
        grad = out_grad.copy()
        grad[self.input_data <= 0] *= self.alpha * np.exp(self.input_data[self.input_data <= 0])
        return grad

#####损失层
        
class SoftmaxCrossEntropyLossLayer(Layer): 
    def forward(self, input_data):
        input_max = np.max(input_data, axis=1, keepdims=True)
        input_exp = np.exp(input_data - input_max)
        self.prediction = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prediction
    def loss(self, y):
        self.batch_size = self.prediction.shape[0]
        self.target = y
        self._loss = -np.sum(np.log(self.prediction) * self.target) / self.batch_size
        return self._loss
    def backward(self): 
        grad = (self.prediction - self.target) / self.batch_size
        return grad

class MSELossLayer(Layer):
    #均方误差
    def forward(self, input_data):
        self.prediction = input_data
        return input_data

    def loss(self, y):
        self.target = y
        self._loss = np.mean(np.sum((self.prediction - self.target)**2, axis=1)) / 2
        return self._loss

    def backward(self):
        grad = (self.prediction - self.target) / self.target.shape[0]
        return grad

class HuberLossLayer(Layer):
    #平滑平均绝对误差
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, input_data):
        self.prediction = input_data
        return input_data

    def loss(self, y):
        self.target = y
        abs_diff = np.abs(self.prediction - self.target)
        quadratic = np.minimum(abs_diff, self.delta)
        self._loss = 0.5 * np.mean(np.sum(quadratic**2, axis=1)) + self.delta * np.mean(np.sum(abs_diff - quadratic, axis=1))
        return self._loss

    def backward(self):
        diff = self.prediction - self.target
        grad = np.where(np.abs(diff) <= self.delta, diff, self.delta * np.sign(diff)) / self.target.shape[0]
        return grad
