import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

datapath='kmeansnonlindata.mat'
matdata=sio.loadmat(datapath)
data=matdata['X']
size=len(data)
x=[data[i][0] for i in range(size)]
y=[data[i][1] for i in range(size)]
target=np.zeros(size)
target[0]=1
t=0
#Initialized
sigma=0.1
plt.subplot(121)
plt.scatter(x,y)

def sum2(x,y):
	return sum((x-y)**2) #Euclidean distance

W=np.zeros([size,size])
m=np.array(data)
for i in range(size):
	for j in range(size):
		W[i][j]=np.exp(-sum2(m[i],m[j])/(2*sigma**2)) #Gaussian Kernel

one=np.matrix([1]*size).T
D=np.diag([float(x) for x in np.dot(W,one)])
L=D-W
L_lam,L_vec=np.linalg.eig(L)
x1=[L_vec[i][0] for i in range(size)]
y1=[L_vec[i][1] for i in range(size)]
data_2d=np.array([x[:2] for x in L_vec])

#Using K-means
m1,m2=data_2d[0],data_2d[1] #initialized
while(t<100):
	target_temp=target.copy()
	for i in range(size):
		m=[sum2(data_2d[i],m1),sum2(data_2d[i],m2)]
		target[i]=m.index(min(m))
	if (target==target_temp).all():
		break
	#update m
	m1=sum(data_2d[target==0])/len(data_2d[target==0])
	m2=sum(data_2d[target==1])/len(data_2d[target==1])
	t+=1

plt.subplot(122)
plt.scatter(x,y,c=target)
plt.show()
