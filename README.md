# a4
Skeleton code for Assignment 4

K Nearest Neighbors

KNN works by finding the distances between a query and all the examples in the data, selecting the specified number examples (K) closest to the query, then votes for the most frequent label (in the case of classification) or averages the labels (in the case of regression).

Methodology:
This program works by calculating the distances between test data and training points. Then selects the k=int number of data points nearest to the test data. Then the program calculates the probability of the test data belonging to the classes of training data and which ever has the highest probability the data points are then classified to that group.

Brief Description on How Program Works:
1) The __init__() method initializes all the class attributes
2) The fit() method sets the parameters equal to the training and test data
3) The predict() method predicts the target values for the test data 
4) The _predict() method calculates the distances, retrieves the k-nearest samples and labels, finds the most common label, and return that label


Multi-layered Perceptron

Methodology:
This program was formulated by first understanding that a multi-layered perceptron is simply a combination of single-layered perceptrons. 
Each perceptron will calculate the probability by multiplying the inputs by the weights and adding bias. This value will then be placed into the activation function. Then these outputs will be combined to new perceptron(s) with new weights and bias'. The final layers output becomes the input for the output activation function. Backpropogation and gradient descent are then used to predict the target values.


Brief Description on How Program Works:
1) The __init__() method is ran to initial all the attributes of the class
2) The _initialize() method is ran at the beginning of the fit() method and executes one hot encoding for the target values. Then it initializes the weights and bias'
3) The fit() method fits the model to the numpy arrays X and y. It also keeps track of the cross entropy loss every 20 iterations. In this method we also calculate  the dot product of the input layer, run hidden_input_layer through hidden activation function, calculate the dot product of the output layer, run output_layer through _output_activation function, calculate the gradients, and update the weights with the learning rates. 
4) The predict() method simply predicts the target class values by using the fitted classifier model. 

Problems Encountered: 
1) sigmoid function has created a lot of slow processing as well as overflow runtime warnings. I have tried implementing the sigmoid function using tanh instead. This removed the errors but runtime is still very slow. So slow that I left my computer for 2 hours while the program ran but only the Iris dataset completed. The Digits dataset seems to be taking very long. This problem may be able to be sped up with a GPU. I believe there's too much computation this neural network needs computed that a CPU just cannot do on it's own. 
