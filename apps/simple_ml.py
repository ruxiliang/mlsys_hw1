"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    
    # parse image file
    with gzip.open(image_filename) as image_file:
        image_file_content = image_file.read()
        pixels = struct.unpack(f">iiii{''.join(['B' for _ in range(len(image_file_content) - 16)])}", image_file_content)
        pixels_ndarray = np.asarray(pixels[4:], dtype=np.float32).reshape(pixels[1],784)
        pixels_ndarray = (pixels_ndarray - pixels_ndarray.min()) / pixels_ndarray.max()
    with gzip.open(label_filename) as label_file:
        label_file_content = label_file.read()
        labels = struct.unpack(f">ii{''.join(['B' for _ in range(len(label_file_content) - 8)])}", label_file_content)
        labels = np.asarray(labels[2:], dtype=np.uint8)
    return pixels_ndarray, labels

    # ### BEGIN YOUR CODE
    # Z_normalized = Z - np.max(Z, axis=1, keepdims=True)
    # logits = np.exp(Z_normalized)
    # A = np.log(np.sum(logits, axis=1))
    # B = Z[np.arange(Z.shape[0]), y]
    # loss = np.log(np.sum(logits, axis=1)) - Z_normalized[np.arange(Z_normalized.shape[0]), y]
    # return loss.sum() / loss.shape[0]
    # ### END YOUR CODE

def softmax_loss(Z: "ndl.Tensor[np.float32]", y_one_hot: "ndl.Tensor[np.int8]"):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    logits = ndl.exp(Z)
    A: ndl.Tensor = ndl.log(ndl.summation(logits, axes=(1,)))
    return (A.sum() - (Z * y_one_hot).sum()) / Z.shape[0]
    
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ## BEGIN YOUR SOLUTION
    test = ndl.Tensor(2.0)
    batch_size = X.shape[0]
    y_onehot = np.zeros((batch_size,y.max() + 1))
    y_onehot[np.arange(batch_size), y] = 1
    for idx in range(0, batch_size, batch):
        X_batch = ndl.Tensor(X[idx:idx+batch, :])
        y_batch = ndl.Tensor(y_onehot[idx:idx+batch, :])
        Z = ndl.matmul(ndl.relu(ndl.matmul(X_batch, W1)), W2)
        loss = softmax_loss(Z, y_batch)
        loss.backward()
        W1 = (W1 - lr * W1.grad).detach()
        W2 = (W2 - lr * W2.grad).detach()
    return (W1, W2)

    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
