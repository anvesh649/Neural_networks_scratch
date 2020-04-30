import numpy as np

from numpy import genfromtxt

data=genfromtxt("DATA/bank_note_data.txt",delimiter=",")
labels=data[:,4]

features=data[:,:4]

X=features
Y=labels

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=42)

from sklearn.preprocessing import MinMaxScaler  #normalize

scaler_object=MinMaxScaler()

scaler_object.fit(x_train)

scaled_x_train=scaler_object.transform(x_train)

scaled_x_test=scaler_object.transform(x_test)	

def initialize_parameters(n_X,n_H,n_Y):
	W1=np.random.randn(n_H,n_X)*0.01
	b1=np.random.randn(n_H,1)
	W2=np.random.randn(n_Y,n_H)*0.01
	b2=np.random.randn(n_Y,1)
	return W1,b1,W2,b2

def nn_model(X, Y, n_h, epochs,learning_rate):
	n_X=X.shape[0]
	n_Y=Y.shape[0]
	m=Y.shape[1]
	def sigmoid(z):
		   return 1/(1+np.exp(-z))
	def tanh(z):
		   return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
	W1,b1,W2,b2=initialize_parameters(n_X,n_h,n_Y)
	for i in range(epochs):
		#forward propagation
		Z1=np.dot(W1,X)+b1
		A1=tanh(Z1)
		Z2=np.dot(W2,A1)+b2
		A2=sigmoid(Z2)
		#cost function
		logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
		cost=-np.sum(logprobs)/m
		#backpropagation
		dZ2 = A2-Y
		dW2 = np.dot(dZ2,A1.T)/m
		db2 = np.sum(dZ2,axis=1,keepdims=True)/m
		dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
		dW1 = np.dot(dZ1,X.T)/m
		db1 =np.sum(dZ1,axis=1,keepdims=True)/m
		#update parameters
		W1 = W1-learning_rate*dW1
		b1 = b1-learning_rate*db1   
		W2 = W2-learning_rate*dW2
		b2 = b2-learning_rate*db2
		print ("Cost after iteration %i: %f" %(i, cost))   

	return A2
y_train=y_train.reshape(919,1)   

k=nn_model(scaled_x_train.T,y_train.T,30,100,0.01)