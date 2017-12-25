import numpy as np
from sklearn import datasets
import pdb

irs = datasets.load_iris()
data = irs.data
target = irs.target

class BP(object):
    def __init__(self, lrate):
        self.lrate = lrate
        self.x_len = 0
        self.y_len = 0

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_p(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))
        
    def set_xy_len(self, x_len, y_len):
        self.x_len = x_len
        self.y_len = y_len

    def train(self, hid_dim, data, loops=20000):
        W1 = np.random.randn(hid_dim, self.x_len)
        b1 = np.random.randn(hid_dim, 1)
        W2 = np.random.randn(self.y_len, hid_dim)
        b2 = np.random.randn(self.y_len, 1)
        #print 'W1:', W1
        #print 'b1:', b1
        #print 'W2:', W2
        #print 'b2:', b2
        
        for i in xrange(loops):
            for X in data:
                pdb.set_trace()
                z1 = X.dot(W1.T) + b1
                break

            #print X, z1
            break

if __name__ == '__main__':
    bp = BP(0.2)
    bp.set_xy_len(4, 3)
    bp.train(10, data)
