#Gaussian kernel
from sklearn.datasets import load_iris
import numpy as np
import random

iris = load_iris()
iris_data = iris['data']
iris_target = iris['target']

#initialized
target=np.zeros(150)
target[0:10]=[1 for x in range(10)]
target[30:60]=[2 for x in range(30)]
t=0
sigma=1

def sum2(x,y):
	return sum((x-y)**2) #Euclidean distance

def inproduct(k,kernel,target,cat):
	catindex=np.array(range(150))[target==cat]
	temp1=sum([kernel[k][x] for x in catindex])
	temp2=sum([kernel[x][y] for x in catindex for y in catindex])
	lens=len(target[target==cat])
	return kernel[k][k]-2*temp1/lens+temp2/lens**2

G=np.zeros([150,150])
for i in range(150):
	for j in range(150):
		G[i][j]=np.exp(-sum2(iris_data[i],iris_data[j])/(2*sigma**2))

while(t<1000):
	target_temp=target.copy()
	for i in range(150):
		m=[inproduct(i,G,target,0),inproduct(i,G,target,1),inproduct(i,G,target,2)]
		target[i]=m.index(min(m))
	if (target==target_temp).all():
		break
	#update m
	t+=1

cat1=target[:50]
cat2=target[50:100]
cat3=target[100:150]
s1,s2,s3=np.median(cat1),np.median(cat2),np.median(cat3)
acc=1-(len(cat1[cat1!=s1])+len(cat2[cat2!=s2])+len(cat3[cat3!=s3]))/150


print(target)
print('iteration:%d' %t)
print('Accuracy:%.2f%%' %(acc*100))