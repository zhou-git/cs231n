import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.s

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
    dW = np.zeros(W.shape) # initialize the gradient as zero
    dW1 = np.zeros(W.shape)
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        counter = 0
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                counter += 1
                dW1[:,j] = X[i]
            else:
                dW1[:,j] = 0
        dW1[:,y[i]] = (-1)*counter*X[i]
        dW += dW1
    dW = dW / num_train

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
      Structured SVM loss function, vectorized implementation.

      Inputs and outputs are the same as svm_loss_naive.
      """
    loss = 0.0
    num_t = X.shape[0]
    dW = np.zeros(W.shape) # initialize the gradient as zero
    #print(X.shape,W.shape)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    loss1 = X.dot(W)
    currect_loss = np.choose(y,loss1.T).reshape(num_t,1)
    y1=y.reshape(num_t,1)
    current_loss_m = np.tile(currect_loss,[1,W.shape[1]])
    loss1 = loss1 - currect_loss +1
    mask = loss1 < 0
    loss1[mask] = 0
    loss = (np.sum(loss1)-num_t*1)/num_t #correct labels having loss 1, should be 0.
    loss += reg * np.sum(W * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    mask_y = loss1 == 1
    loss1[mask_y] = 0
    loss1[loss1 > 0] = 1
    sum_loss = np.sum(loss1,1)
    loss1[np.arange(num_t), y] = -sum_loss
    #print(loss1[[1,2,3],[2,4,7]])
    dW = X.T.dot(loss1)
    dW /= num_t
    dW += reg * W
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################

    return loss, dW
