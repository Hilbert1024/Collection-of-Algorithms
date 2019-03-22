# Support Vector Machine
## Linear SVM
### Hard-margin
If the training data is linearly separable, we can select two parallel hyperplanes that separate the two classes of data, so that the distance between them is as large as possible.
### Soft-margin
If the training data is not linearly separable, for data on the wrong side of the margin, the function's value is proportional to the distance from the margin.
## Kernel SVM
We use kernel trick to deal with nonlinear classification problem.
## SMO
Sequential minimal optimization (SMO) is an algorithm for solving the quadratic programming (QP) problem that arises during the training of support vector machines (SVM).
## Files
svm_kernel : With Gaussian kernel and SMO algorithm, the results is 
+ train accuracy: 95.0%
+ test accuracy: 95.5%
