import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
                  Softmax loss function, naive implementation (with loops)

                  Inputs have dimension D, there are C classes, and we operate on minibatches
                  of N examples.

                  Inputs:
                  - W: A numpy array of shape (D, C) containing weights.
                  - X: A numpy array of shape (N, D) containing a minibatch of data.
                  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
                    that X[i] has label c, where 0 <= c < C.
                  - reg: (float) regularization strength

                  Returns a tuple of:
                  - loss as single float
                  - gradient with respect to weights W; an array of same shape as W
                  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_class = dW.shape[1]
    score  = X.dot(W)
    C=np.reshape(np.max(score, axis = 1), (num_train,1))
    #print(C.shape,score.shape,num_class)
    score -= np.tile(C, (1, num_class))
    #print(score)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    marginal = np.sum(np.exp(score),axis = 1)
    t=score[np.arange(num_train),y]
    mar1 = np.reshape(marginal, (num_train,1))
    tt = np.exp(score)/ np.tile(mar1,(1,num_class))
    tt[np.arange(num_train), y] -= 1
    loss1 =-np.log(np.exp(t) /marginal)
    dW = X.T.dot(tt) /num_train

    loss = np.sum(loss1)/num_train
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_class = dW.shape[1]
    score  = X.dot(W)
    C=np.reshape(np.max(score, axis = 1), (num_train,1))
    #print(C.shape,score.shape,num_class)
    score -= np.tile(C, (1, num_class))
    #print(score)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    marginal = np.sum(np.exp(score),axis = 1)
    t=score[np.arange(num_train),y]
    mar1 = np.reshape(marginal, (num_train,1))
    tt = np.exp(score) / np.tile(mar1,(1,num_class))
    tt[np.arange(num_train), y] -= 1
    loss1 =-np.log(np.exp(t) /marginal)
    dW = X.T.dot(tt)/num_train

    loss = np.sum(loss1)/num_train
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW