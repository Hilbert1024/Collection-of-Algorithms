from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
iris_data = iris['data']
iris_target = iris['target']

m1,m2,m3=iris_data[0],iris_data[70],iris_data[140] #initialized
target=np.zeros(150)
t=0
def sum2(x,y):
	return sum((x-y)**2) #Euclidean distance

while(t<1000):
	target_temp=target.copy()
	for i in range(150):
		m=[sum2(iris_data[i],m1),sum2(iris_data[i],m2),sum2(iris_data[i],m3)]
		target[i]=m.index(min(m))
	if (target==target_temp).all():
		break
	#update m
	m1=sum(iris_data[target==0])/len(iris_data[target==0])
	m2=sum(iris_data[target==1])/len(iris_data[target==1])
	m3=sum(iris_data[target==2])/len(iris_data[target==2])
	t+=1

cat1=target[:50]
cat2=target[50:100]
cat3=target[100:150]
s1,s2,s3=np.median(cat1),np.median(cat2),np.median(cat3)
acc=1-(len(cat1[cat1!=s1])+len(cat2[cat2!=s2])+len(cat3[cat3!=s3]))/150

print(target)
print('iteration:%d' %t)
print('Accuracy:%.2f%%' %(acc*100))