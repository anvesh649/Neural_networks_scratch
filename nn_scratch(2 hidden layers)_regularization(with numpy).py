import numpy as np
def initialize_parameters(n_X,n_H,n_Y):
	W1=np.random.randn(n_H,n_X)
	b1=np.random.randn(n_H,1)
	W2=np.random.randn(n_H,n_H)
	b2=np.random.randn(n_H,1)
	W3=np.random.randn(n_Y,n_H)
	b3=np.random.randn(n_Y,1)
	return W1,b1,W2,b2,W3,b3

def nn_model(X, Y, n_h, epochs,learning_rate,lambd):
	n_X=X.shape[0]
	n_Y=Y.shape[0]
	m=Y.shape[1]
	def sigmoid(z):
		   return 1/(1+np.exp(-z))
	def tanh(z):
		   return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
	W1,b1,W2,b2,W3,b3=initialize_parameters(n_X,n_h,n_Y)
	for i in range(epochs):
		#forward propagation
		Z1=np.dot(W1,X)+b1
		A1=tanh(Z1)
		Z2=np.dot(W2,A1)+b2
		A2=tanh(Z2)
		Z3=np.dot(W3,A2)+b3
		A3=sigmoid(Z3)
		#cost function
		logprobs = np.multiply(np.log(A3), Y) + np.multiply((1 - Y), np.log(1 - A3))
		cost=-np.sum(logprobs)/m
		L2_reg_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)
		cost1 = cost + L2_reg_cost    

		#backpropagation
		dZ3 = A3-Y
		dW3 = np.dot(dZ3,A2.T)/m+ lambd*(W3/m)
		db3 = np.sum(dZ3,axis=1,keepdims=True)/m
		dZ2 = np.dot(W3.T,dZ3)*(1-np.power(A2,2))
		dW2 = np.dot(dZ2,A1.T)/m+ lambd*(W2/m)
		db2 = np.sum(dZ2,axis=1,keepdims=True)/m
		dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))	
		dW1 = np.dot(dZ1,X.T)/m+ lambd*(W1/m)
		db1 =np.sum(dZ1,axis=1,keepdims=True)/m
		#update parameters
		W1 = W1-learning_rate*dW1
		b1 = b1-learning_rate*db1   
		W2 = W2-learning_rate*dW2
		b2 = b2-learning_rate*db2
		W3 = W3-learning_rate*dW3
		b3 = b3-learning_rate*db3
		print ("Cost after iteration %i: %f" %(i, cost1))   
			
	return A3
