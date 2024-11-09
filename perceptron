import numpy as np

def unit_step_func(x):
    return np.where(x>0, 1, 0)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=100):
        self.lr = learning_rate #alpha basically. learning rate coefficeint for perceptron update rule
        self.n_iters = n_iters #n_ matlab number of. number of iterations over the given data you want to take. More the iterations, the more the model gets time to learn and better the results.
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None

    #fit function: yeh weight and bias ko iteratively seekhta(closer to answer banata) hai 
    def fit(self, x, y):
        # x is numpy array of training data, y is the numpy array of answers i.e labels
        #labels array mein samples hain. har sample bhi ek array hai jismein instances hain
        #sample/instance: ek observation/datapoint ex: hous
        #feature: ek measurable property or characteristic of the sample ex: size, number of bedrooms
        
        #x is a numpy array and .shape attribute gives uska dimensions i.e the number of samples and features, as a tuple.
        n_samples, n_features = x.shape
        
        self.weights = np.zeros(n_features) #initialising the weights at first as 0s
        self.bias = 0#same for bias

        y_ = np.where(y>0, 1, 0) #to transform the answers(labels) into a binary 0 or 1 format so that we can continue the training


        #now the weight learning part
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(x):#for index and value in x
                #The enumerate function in Python is used to iterate over a sequence (such as a list or array) while keeping track of both the index and the value of each element in the sequence. It returns an iterator that produces pairs of index and value
                linear_output = np.dot(x_i, self.weights) + self.bias #just applying the formula for the approximate result now
                y_predicted = self.activation_func(linear_output) #approximate ka final output

                #now applying the "perceptron update rule"
                update = self.lr*(y_[idx] - y_predicted) 
                self.weights += update*x_i #del(w) = alpha.(y_actual - y_pred) * x_i
                self.bias += update #del(b) = alpha.(y_actual - y_pred)

    #now to actually get the answer using the updated and learned values of weights and bias by appyling the same formula but with learned values
    def predict(self, x):
        #notice that the answer array: y, is absent this time, for obvs reasons
        linear_output = np.dot(x, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
    
#This class is now a single perceptron - neuron ka mimicry for ml.
#we can now create instances of these and use them.

#you can now test this with some data from maybe matplotlib.
