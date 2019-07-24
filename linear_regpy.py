import csv
import numpy as np
from collections import defaultdict
columns = defaultdict(list)
theta = []
with open('kanay.csv')as f:
	reader = csv.DictReader(f)
	for row in  reader:
		for(k,v) in row.items():
			columns[k].append(v)
m = len(columns[0])
z = np.full((m,1),1)
p = [[columns[0]],[columns[1]],[ columns[2]]]
x = np.append(z,p)
y =np.array([3]).reshape(m,1)
print(x)
print(y)
#=[a1] , [a2] ,[a3]]
a =np.array([[a0] , [a1] , [a2] , a[3]])
def regression():
	''' i in range(len(x)):
		for j in range(len(y[0])):
			for k in range(len(y)):
				result[i][j] += x[i][k]*y[k][j]'''
	func = np.dot(x,a)
	sub = np.subtract(func,y)
	matrix = np.square(sub)
	ffunc = np.sum(matrix)
	return (1/(2*m))*ffunc
def gradient():
	for i in range (1,1500) :
		sub = np.subtract(func,y)
		theta = theta -(0.01/m)*(x.T*sub)
	    
	 



