import numpy as np
import math
import itertools

#******layers******

class Layer(object):
	"""Base class of all layer objects"""
	def forward(self, input):  
		"""calculate activations for this layer"""
		pass
		
	def backward(self, input=None):
		"""Gradient of this layer wrt inputs ie. d(outputs)/d(inputs)"""
		pass

	def delta(self, input=None, delta_next=None,):
		"""Cumulative gradient up to the inputs of this layer
		delta_next - gradient up to the outputs of this layer"""
		return delta_next.dot(self.backward())

	def get_params_grad(self, input=None, delta_next=None):
		"""Gradient wrt the weights and biases of this layer"""
		return []
		
	def get_params_grad_matrix(self, input, delta_next):
		"""Gradient wrt the weights and biases of this layer"""
		return [], []
		
	def get_params_iter(self):
		"""Return an iterator over the parameters (if any).
		The iterator has the same order as get_params_grad.
		The elements returned by the iterator are editable in-place."""
		return []


class Linear(Layer):
	"""Applies a linear transformation to the incoming data: y = xW + b"""
	def __init__(self, n_in=1, n_out=1,W_given=None,b_given=None):
		if W_given is not None:
			self.n_in = W_given.shape[0]
			self.n_out = W_given.shape[1]
			self.W = W_given
		else:
			self.n_in = n_in
			self.n_out = n_out
			stdv = 1. / math.sqrt(n_in)
			self.W = np.random.randn(n_in,n_out).astype(np.longdouble) * np.sqrt(2/n_in) #2 * stdv * np.random.random_sample((n_in, n_out)).astype(np.longdouble) - stdv #uniform init
		if b_given is not None:
			self.b = b_given
		else:
			self.b = np.zeros(n_out).astype(np.longdouble) #zero init
		
	def __str__(self):
		return 'Linear({},{})'.format(self.n_in,self.n_out)
			
	def forward(self, input):  
		"""calculate activations for this layer"""
		return input.dot(self.W) + self.b
		
	def backward(self):
		"""Gradient of this layer wrt its inputs ie. dy/dx"""
		return self.W.T 
		
	def get_params_grad(self, input, delta_next):
		"""Gradient wrt the weights and biases of this layer"""
		dW = input.T.dot(delta_next)
		db = np.sum(delta_next, axis=0)
		return [g for g in itertools.chain(np.nditer(dW), np.nditer(db))]
		
	def get_params_grad_matrix(self, input, delta_next):
		"""Gradient wrt the weights and biases of this layer"""
		dW = input.T.dot(delta_next)
		db = np.sum(delta_next, axis=0)
		return dW , db

	def get_params_iter(self):
		"""Return an iterator over the parameters."""
		return itertools.chain(np.nditer(self.W, op_flags=['readwrite']),
								np.nditer(self.b, op_flags=['readwrite']))

		
class Relu(Layer):
	def __str__(self):
		return 'ReLU()'
		
	def forward(self, input):
		input[input < 0] = 0
		return input

	def backward(self, input):
		input[input < 0] = 0
		input[input > 0] = 1
		return input
		
	def delta(self, input, delta_next):
		return delta_next * self.backward(input)
		

#******loss functions******		

class CrossEntropyWithSoftmax:
	"""The softmax output layer computes the classification propabilities at the output."""		
	def loss(self, input, y_true):
		m = y_true.shape[0]
		y = y_true.argmax(axis=1)
		sm = softmax(input)
		log_likelihood = -np.log(sm[range(m),y])
		loss = np.sum(log_likelihood) / m
		return loss
		
	def backward(self, input, y_true):
		num_examples = input.shape[0]
		y = y_true.argmax(axis=1)
		probs = softmax(input)
		probs[range(num_examples), y] -= 1
		return probs
        
	def delta(self, y_pred, y_true):
		return self.backward(y_pred, y_true)
		
		
class MSE:
	"""Calculate mean square error between predictions and targets"""
	def loss(self, target, pred):
		return np.mean((pred - target)**2)

	def backward(self, y_pred, y_true):
		return y_pred - y_true
		
	def delta(self, y_pred, y_true):
		return self.backward(y_pred, y_true)


#******functional******

def softmax(input):
	"""Calculate (stable) softmax."""
	e_x = np.exp(input - np.max(input))
	return e_x / np.sum(e_x, axis=1, keepdims=True) 