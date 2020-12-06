# https://medium.com/@a.mirzaei69/implement-a-neural-network-from-scratch-with-python-numpy-backpropagation-e82b70caa9bb

# I'm not sure if this is going to work with more than 1 feature

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(object):
    def __init__(self, layers, activations):
        assert(len(layers) == len(activations)+1)
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i+1], layers[i]))
            self.biases.append(np.random.randn(layers[i+1], 1))

    def feedforward(self, x):
        # return the feedforward value for x
        a = np.copy(x)
        z_s = []
        a_s = [a]
        for i in range(len(self.weights)):
            activation_function = self.getActivationFunction(self.activations[i])
            z_s.append(self.weights[i].dot(a) + self.biases[i])
            a = activation_function(z_s[-1])
            a_s.append(a)
        return (z_s, a_s)

    def backpropagation(self,y, z_s, a_s):
        dw = []  # dC/dW
        db = []  # dC/dB
        deltas = [None] * len(self.weights)  # delta = dC/dZ  known as error for each layer
        # insert the last layer error
        deltas[-1] = ((y-a_s[-1])*(self.getDerivitiveActivationFunction(self.activations[-1]))(z_s[-1]))
        # Perform BackPropagation
        for i in reversed(range(len(deltas)-1)):
            deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*(self.getDerivitiveActivationFunction(self.activations[i])(z_s[i]))
        #a= [print(d.shape) for d in deltas]
        batch_size = y.shape[1]
        db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas]
        dw = [d.dot(a_s[i].T)/float(batch_size) for i,d in enumerate(deltas)]
        # return the derivitives respect to weight matrix and biases
        return dw, db

    def train(self, x, y, batch_size=10, epochs=100, lr = 0.01):
        # update weights and biases based on the output
        for e in range(epochs):
            i=0
            while(i<len(y)):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                i = i+batch_size
                z_s, a_s = self.feedforward(x_batch)
                dw, db = self.backpropagation(y_batch, z_s, a_s)
                self.weights = [w+lr*dweight for w,dweight in  zip(self.weights, dw)]
                self.biases = [w+lr*dbias for w,dbias in  zip(self.biases, db)]
                print("loss = {}".format(np.linalg.norm(a_s[-1]-y_batch) ))

    @staticmethod
    def getActivationFunction(name):
        if(name == 'sigmoid'):
            return lambda x : 1/(1+np.exp(-1*x))

        elif(name == 'hyperbolic-tangent'):
            return lambda x : ( np.exp(x) - np.exp(-1*x) ) / ( np.exp(x) + np.exp(-1*x) )

        else:
            error ("UNKNOWN ACTIVATION FUNCTION")
            quit()

    @staticmethod
    def getDerivitiveActivationFunction(name):
        if(name == 'sigmoid'):
            sig = lambda x : 1/(1+np.exp(-1*x))
            return lambda x :sig(x)*(1-sig(x))
        elif(name == 'hyperbolic-tangent'):
            tanh = lambda x : ( np.exp(x) - np.exp(-1*x) ) / ( np.exp(x) + np.exp(-1*x) )
            return lambda x : 1 - tanh(x) ** 2
        else:
            error ("UNKNOWN ACTIVATION FUNCTION")
            quit()



if __name__=='__main__':
    nn = NeuralNetwork([1, 4],activations=['hyperbolic-tangent'])

    # Read in data
    X = np.array([])
    y = np.array([])
    dataFile = open('data_banknote_authentication.txt', 'r')
    fullData = dataFile.readlines()
    for row in fullData:
        data = row.strip().split(",")
        newX = np.array([])
        #for d in data[:-1]:
            #newX = np.append ( newX, float(d) )
        newX = np.array( [float( data[0] )] )
        # FIXME use other features
        X = np.append ( X, newX )
        y = np.append ( y, float(row.strip().split(",")[-1]) )

    X = np.array([X])
    y = np.array([y])

    '''
    X = 2*np.pi*np.random.rand(1000).reshape(1, -1)
    y = np.sin(X)
    '''

    print ("X", X)
    print ("y", y)

    nn.train(X, y, epochs=10, batch_size=64, lr = .1)
    _, a_s = nn.feedforward(X)

    #print(y, X)
    '''
    plt.scatter(X.flatten(), y.flatten())
    plt.scatter(X.flatten(), a_s[-1].flatten())
    plt.show()
    '''
