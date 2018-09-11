import numpy as np
import collections
import itertools


#******training routines******

def feedforward(input, layers):
	"""Compute and return the forward activation of each layer in layers."""
	activations = [input] 
	X = input
	for layer in layers:
		Y = layer.forward(X)  #feed the input into the layer
		activations.append(Y) 
		X = Y  #use the current output as the input for the next layer 
	return activations

	
def backprop(activations, targets, layers, loss_func):
	"""Perform backpropagation over all layers and return the parameter gradients.
	Parameters:
		activations - A list of forward step activations (outputs of each layer)
			activations[-1] contains the predictions.
			activations[0] contains the input samples. 
		targets - The training targets (layers)
		layers - A list of Layers constituting the model
		loss_func - the loss function to evaluate the output eg. MSE, NLL
	Return:
		A list of parameter gradients where the gradients at each index corresponds to
		the parameters gradients of the layer at the same index in layers. 
	"""
	param_grads = collections.deque()  # List of parameter gradients for each layer
	Y = activations.pop()
	input_grad = loss_func.delta(Y, targets) #First get the gradient from the loss function
	output_grad = input_grad
	for layer in reversed(layers): #iterate trhough layers backwards
		Y = activations.pop()
		input_grad = layer.delta(Y, output_grad)
		grads = layer.get_params_grad(Y, output_grad)
		param_grads.appendleft(grads)
		output_grad = input_grad
	return list(param_grads)

def backprop_return_grads(activations, targets, layers, loss_func):
	"""Perform backpropagation over all layers and return the parameter gradients.
	Parameters:
		activations - A list of forward step activations (outputs of each layer)
			activations[-1] contains the predictions.
			activations[0] contains the input samples. 
		targets - The training targets (layers)
		layers - A list of Layers constituting the model
		loss_func - the loss function to evaluate the output eg. MSE, NLL
	Return:
		A list of parameter gradients where the gradients at each index corresponds to
		the parameters gradients of the layer at the same index in layers. 
		Derivatives of eights and biases
	"""
	param_grads = collections.deque()  # List of parameter gradients for each layer
	Y = activations.pop()
	input_grad = loss_func.delta(Y, targets) #First get the gradient from the loss function
	output_grad = input_grad
	dW_all , db_all = [], []
	for layer in reversed(layers): #iterate trhough layers backwards
		Y = activations.pop()
		input_grad = layer.delta(Y, output_grad)
		grads = layer.get_params_grad(Y, output_grad)
		dW, db = layer.get_params_grad_matrix(Y, output_grad)
		dW_all.append(dW)
		db_all.append(db)
		param_grads.appendleft(grads)
		output_grad = input_grad
	return list(param_grads), dW_all[::-1], db_all[::-1]

def update_params(layers, param_grads, learning_rate):
    """Update the model parameters using SGD"""
    for layer, layer_backprop_grads in zip(layers, param_grads):
        for param, grad in zip(layer.get_params_iter(), layer_backprop_grads):
            # The parameter returned by the iterator point to the memory space of
            #  the original layer and can thus be modified inplace.
            param -= learning_rate * grad

			
#******processing routines******

def create_minibatch(dataset,labels,batch_size):
    n_batches = dataset.shape[0] / batch_size  # Number of batches
    train_batches = zip(
        np.array_split(dataset, n_batches, axis=0),  # X samples
        np.array_split(labels, n_batches, axis=0))  # Y targets
    return train_batches, n_batches
	
			
def shuffle(dataset, labels):
    """Randomizes order of elements in input arrays"""
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:]
    shuffled_labels = labels[permutation,:]
    return shuffled_dataset, shuffled_labels

	
def label2onehot(labels,n_labels):
    """1-hot encoding for labels. n_labels is the no. of classes"""
    oh_labels = (np.arange(n_labels) == labels[:]).astype(np.int32)
    return oh_labels
	
	
def accuracy(model_outputs, labels):
	pred = softmax(model_outputs)
	correct_pred = np.equal(pred.argmax(axis=1), labels.argmax(axis=1))
	accuracy = 100*np.mean(correct_pred)
	return accuracy

	
#******utility functions******

def time_taken(elapsed):
    """To format time taken in hh:mm:ss. Use with time.monotic()"""
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)
	
def softmax(input):
	"""Calculate (stable) softmax."""
	e_x = np.exp(input - np.max(input))
	return e_x / np.sum(e_x, axis=1, keepdims=True) 
