#Gaussian kernel
import numpy as np
import math
import matplotlib.pyplot as plt

size1=100
r1=np.random.uniform(0,1,size1)
theta1=np.random.uniform(0,2*math.pi,size1)

size2=200
r2=np.random.uniform(1.5,2,size2)
theta2=np.random.uniform(0,2*math.pi,size2)

size=size1+size2
x1=r1*np.sin(theta1)
y1=r1*np.cos(theta1)

x2=r2*np.sin(theta2)
y2=r2*np.cos(theta2)

x=np.append(x1,x2)
y=np.append(y1,y2)

data=[]
for i in range(size1):
	data.append([x1[i],y1[i]])
for i in range(size2):
	data.append([x2[i],y2[i]])
data=np.array(data)

#initialized
target=np.zeros(size)
target[0]=1
t=0

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
		G[i][j]=np.exp(-sum2(data[i],data[j])/2)

while(t<1000):
	target_temp=target.copy()
	for i in range(size):
		m=[inproduct(i,G,target,0,size),inproduct(i,G,target,1,size)]
		target[i]=m.index(min(m))
	if (target==target_temp).all():
		break
	#update m
	t+=1

plt.scatter(x,y,c=target)
# cat1=target[:size1]
# cat2=target[size1:]

# s1,s2=np.median(cat1),np.median(cat2)
# acc=1-(len(cat1[cat1!=s1])+len(cat2[cat2!=s2]))/size


# print(target)
print('iteration:%d' %t)
plt.show()
# print('Accuracy:%.2f%%' %(acc*100))