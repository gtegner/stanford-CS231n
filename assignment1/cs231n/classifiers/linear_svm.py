import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  N = X.shape[0]
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  loss1 = loss

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  scores = np.dot(X,W)
  correct_class_score = scores[range(N),y]

  #print("scores")
  #print(scores[0:3,:])
  #print("correct class scores")
  #print(correct_class_score[0:3])
  #print("y")

  #print(y[0:3])

  ymat = np.tile(correct_class_score,(num_classes,1)).T

  ones = np.ones(np.shape(ymat))

  M = scores-correct_class_score[:,np.newaxis]+ones
  M[M<0] = 0

  #print("M")
  M[range(N),y] = 0

  loss = np.sum(M)
  loss /= N

  #print(M[0:3,:])


  M = scores-correct_class_score[:,np.newaxis]+ones

  M[M<0] = 0
  M[M>0] = 1

  M[range(N),y] = 0
  M[range(N),y] = -1 * np.sum(M,axis=1)
 # print(M)

  dW = np.dot(X.T,M)
  dW /= float(N)
  dW += reg*W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  N = X.shape[0]
  C = W.shape[1]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = np.dot(X,W)
  correct_class_scores = scores[range(N),y]
  L = scores - correct_class_scores[:,np.newaxis] + np.ones((N,C))
  L[range(N),y] = 0
  L[L<0] = 0
  loss = np.sum(L) / float(N)

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

  L = scores - correct_class_scores[:,np.newaxis] + np.ones((N,C))
  L[L<0] = 0
  L[L>0] = 1
  L[range(N),y] = 0
  L[range(N),y] = -1*np.sum(L,axis=1)

  dW = np.dot(X.T,L)
  dW /= float(N)
  dW += reg*W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
