import numpy as np

class LinearRegression():
    def __init__(self):
        #model parameters - theta being weights
        self.theta = None
        self.X = None
        self.Y = None
    
    def matrix_maker(self, features, output, data, m):
        #Create X matrix and 1 matrix
        data_array = data[features].to_numpy()
        array_1s = np.ones([1,m])
        
        #hstack to easily pad leftmost column with array of 1s
        self.X = np.hstack((array_1s.T, data_array))
        self.Y = data[[output]].to_numpy()
                
    
    def fit(self):
        #returns both slope and intercept (w1, w0 respectively) in theta through "fitting"
        A = np.linalg.pinv(np.dot(self.X.T, self.X)) #pseudoinverse
        B = np.dot(self.X.T, self.Y)
        w = np.dot(A,B) # 2x2 - w0 and w1
        self.theta = w
        return self.theta
        
    def predict(self, x_val):
        #returns y val as prediction through theta's weights
        y_val = np.dot(x_val, self.theta)
        return y_val
        
    def cost_function(self, m):
    # x dotted with weights - take actual y and find the difference (residual)
        x = self.X
        y = self.Y
        residual = np.dot(x, self.theta) - y
        cost = (1/(2*m)) * np.dot(residual.T, residual)
        return cost
        