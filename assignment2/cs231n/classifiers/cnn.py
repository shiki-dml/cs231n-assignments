from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.params[f"W{1}"] = np.random.normal(loc= 0,scale = weight_scale,size =(num_filters,input_dim[0],filter_size,filter_size))
        self.params[f"b{1}"] = np.zeros(num_filters)
        self.params[f"W{2}"] = np.random.normal(loc=0,scale=weight_scale,size=(num_filters*input_dim[1]//2*input_dim[2]//2,hidden_dim))
        self.params[f"b{2}"] = np.zeros(hidden_dim)
        self.params[f"W{3}"] = np.random.normal(loc=0,scale=weight_scale,size = (hidden_dim,num_classes))
        self.params[f"b{3}"] = np.zeros(num_classes)
        

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        out_crp,cache_crp =conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
        out_afr,cache_afr = affine_relu_forward(out_crp,W2,b2)
        out_af,cache_af = affine_forward(out_afr,W3,b3)
        scores = out_af
        
        if y is None:
            return scores

        loss, grads = 0, {}
        loss,dx = softmax_loss(out_af,y)
        dx,grads['W3'],grads['b3'] = affine_backward(dx,cache_af)
        dx,grads['W2'],grads['b2'] = affine_relu_backward(dx,cache_afr)
        dx,grads['W1'],grads['b1'] = conv_relu_pool_backward(dx,cache_crp)
        for i in range(1, 4):
          loss += 0.5 * self.reg * np.sum(self.params[f'W{i}'] ** 2)
          grads[f'W{i}'] += self.reg * self.params[f'W{i}']


        return loss, grads
