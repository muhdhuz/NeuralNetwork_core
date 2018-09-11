## NeuralNetwork_core

An implementation of neural network layers from scratch using Numpy.  
Does feedforward and backpropagation to train a deep learning network.  

**Layers class**  
Base class of all layer objects. Each has a forward and backward method, plus a delta value to propagate the gradients.  
Additionally, network layers (e.g. linear) has methods to grab the gradients of the weights and biases.  

Network layers available:

```bash
Linear()
```
Activation layers available:
```bash
ReLu()
```		
**Loss functions**  
Loss functions are implemented similarly to layers class, except their forward method is replaced with a loss calculation.

Loss functions available:
```bash
CrossEntropyWithSoftmax()
MSE()
```
**Using the modules**  
There are also other processing routines available for minibatching, shuffling, converting to one-hot labels etc.  
To create the network first append the desired layers to a list:
```bash
import network_modules as nm

model = []
model.append(nm.Linear(14,100))
model.append(nm.Relu())
model.append(nm.Linear(100,40))
model.append(nm.Relu())
model.append(nm.Linear(40,4))
```
Thereafter, initialize the loss functions and run the training loop:
```bash
import training as tr

criterion = nm.MSE()
for i in range(500):
    
    activations = tr.feedforward(inputs,model) # Get the activations
    loss = criterion.loss(activations[-1],targets)  # Get loss

    param_grads = tr.backprop(activations, targets, model, criterion)  # Get the gradients
    tr.update_params(model, param_grads, learning_rate)  # Update the parameters using SGD
    
    if (i+1) % log_interval == 0:
        print ('Step [{}/{}], Loss: {:.6f}' 
               .format(i+1, 500, loss))
```
**Dependencies**  
* python 3.6  
* numpy 1.14  

**Authors**  
* Muhammad Huzaifah






