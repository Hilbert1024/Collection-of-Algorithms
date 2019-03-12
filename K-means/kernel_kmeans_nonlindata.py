import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

datapath=r'C:\Users\Hilbert\Desktop\大数据算法\kmeans\kmeansnonlindata.mat'
matdata=sio.loadmat(datapath)
data=matdata['X']
size=len(data)
x=[data[i][0] for i in range(size)]
y=[data[i][1] for i in range(size)]
target=np.zeros(size)
target[0]=1
t=0
#Initialized
sigma=1
target_list=[]

def sum2(x,y):
	return sum((x-y)**2) #Euclidean distance

def inproduct(k,kernel,target,cat,size):
	catindex=np.array(range(size))[target==cat]
	temp1=sum([kernel[k][x] for x in catindex])
	temp2=sum([kernel[x][y] for x in catindex for y in catindex])
	lens=len(target[target==cat])
	return kernel[k][k]-2*temp1/lens+temp2/lens**2

G=np.zeros([size,size])
for i in range(size):
	for j in range(size):
		G[i][j]=np.exp(-sum2(data[i],data[j])/(2*sigma**2))

plt.subplot(221)
plt.scatter(x,y,c=target)
while(t<8):
	target_temp=target.copy()
	target_list.append(target_temp)
	for i in range(size):
		m=[inproduct(i,G,target,0,size),inproduct(i,G,target,1,size)]
		target[i]=m.index(min(m))
	if (target==target_temp).all():
		break
	#update m
	t+=1
target_list.append(target)
for i in range(9):
	plt.subplot(331+i)
	plt.scatter(x,y,c=target_list[i])
plt.show()