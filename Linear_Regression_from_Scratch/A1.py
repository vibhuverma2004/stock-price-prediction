#!/usr/bin/env python
# coding: utf-8

# In[25]:
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### 1. Write a function to generate a data matrix X. 
# **Inputs**: Number of samples, feature dimension. 
# 
# **Output**: Data matrix X.
# 

# In[26]:


def matrix_generator(sample_count, feature_dimension):
    matrix = np.random.randint(100, size=(sample_count, feature_dimension))
    return matrix


# ### 2. Write a function to generated dependent variable column t. 
# **Inputs**: Data matrix X, weight vector for each column, bias w0, noise variance
# 
# **Output**: Target vector t

# In[27]:


def column_generator(matrix, weight_vector, bais_w_0, noise_variance):
    noise=np.random.normal(loc=0, scale= noise_variance, size= bais_w_0.shape)
#   weight_vector_transpose = weight_vector.transpose() 
    xw = np.dot( matrix, weight_vector ) # matrix multiplication
    return xw + bais_w_0  + noise


# ### 3. Write a function to compute a linear regression estimate. 
# 
# **Input**: data matrix X and weight vector w
# 
# **Output**: y

# In[28]:


def calculate_estimate_y(matrix, weight_vector):
    return np.dot( matrix, weight_vector ) # matrix multiplication 


# ### 4. Write a function to compute the mean square error of two vectors y and t.
# 

# In[29]:


def mean_square_error(estimate_y, target_y):
    return (np.square(estimate_y - target_y)).mean(axis=0)
    


# ### 5. Write a function to estimate the weights of linear regression using pseudo-inverse, assuming L2 regularization 
# **Input**: X, t, and lambda
# 
# **Output**: w, MSE, y

# In[30]:


def estimate_weight(matrix, target_y , lambda_l2):
    transpose_matrix = matrix.transpose()
    matrix_multiplication = np.dot(transpose_matrix , matrix)
    identity_size = lambda_l2*np.identity(matrix_multiplication.shape[0])
    add_matrix = identity_size + matrix_multiplication
    add_matrix_inverse = np.linalg.inv(add_matrix)
    matrix_multiplication_target_y = np.dot(transpose_matrix, target_y)
    weight = np.dot(add_matrix_inverse , matrix_multiplication_target_y)
    pridicted_y = np.dot(matrix, weight)
    MSE= mean_square_error(pridicted_y, target_y)
    return weight, MSE, pridicted_y                                     #return as tuple  


# ### 6. Write a function to compute the gradient of MSE with respect to its weight vector. 
# **Input**: X matrix, t vector, and w vector
# 
# **Output**: gradient vector

# In[31]:


def  compute_gradient_of_MSE(matrix, t_vector, w_vector):
    x_t=matrix.transpose()
    return 2*np.dot(np.dot(x_t, matrix), w_vector) - 2*np.dot(x_t, t_vector)


# ### 7. Write a function to compute L2 norm of a vector w passed as a numpy array. Exclude bias w0

# In[32]:


def compute_l2_norm(weight_vector):
    return np.linalg.norm(weight_vector, ord=2)


# ### 8. Write a function to compute the gradient of L2 norm with respect to the weight vectors.
# 
# **Input**: X matrix and w vector
# 
# **Output**: gradient vector, where gradient with respect to w0 is 0.

# In[33]:


def  compute_gradient_of_L2_norm(matrix, vector_w):
    return vector_w


# ### 9. Write a function to compute L1 norm of a vector w passed as a numpy array. Exclude bias w0.
# 

# In[34]:


def compute_l1_norm(weight_vector):
    return np.linalg.norm(weight_vector, ord=1)


# ### 10. Write a function to compute the gradient of L1 norm with respect to the weight vectors. 
# 
# a) ***Input***: X matrix and w vector
# 
# b) ***Output***: gradient vector, where gradient with respect to w0 is 0.

# In[35]:


def  compute_gradient_of_L1_norm(matrix, vector_w):
    l =[1]*vector_w.shape[0]
    return np.array(l)


# ### 11. Write a function for a single update of weights of linear regression using gradient descent. 
# a) **Input**: X, t, w, eta, lambda 2, lambda 1. Note that the weight of MSE will be 1
# 
# b) **Output**: updated weight and updated MSE

# In[36]:


def single_update_weight(X, t, w, eta, lambda_1, lambda_2):
    gradient_mse = compute_gradient_of_MSE(X, t, w)
    updated_weight = w - eta * gradient_mse
    new_predicted_y = np.dot(X, updated_weight)
    updated_mse = mean_square_error(new_predicted_y, t)
    return updated_weight, updated_mse


# ### 12. Write a function to estimate the weights of linear regression using gradient descent.
# 
# a) **Inputs**: X, t, lambda2 (default 0), lambda1 (default 0), eta, max_iter, min_change_NRMSE
# 
# b) **Output**: Final w, final RMSE normalized with respect to variance of t.
# 
# c) **Stopping criteria**: Either max_iter has been reached, or the normalized RMSE does not change by more than
# min_change_NRMSE

# In[37]:


def estimate_weight_gradient_descent(X, t, eta, max_iter, min_change_NRMSE, lambda2=0, lambda1=0):
    weight, MSE, pridicted_y = estimate_weight(X, t, lambda2)
    for i in range(max_iter):
        updated_weight, updated_mse = single_update_weight(X, t, weight, eta, lambda1, lambda2)
        nrmse = np.sqrt(updated_mse) / np.mean(t)
        if nrmse < min_change_NRMSE:
            break
    return updated_weight, nrmse


# ### 13. Run multiple experiments (with different random seeds) for, plot the results of (box plots), and comment on the trends and potential reasons for the following relations:

# a) Training and validation NRMSE obtained using pseudo inverse with number of training samples 

# In[38]:


""""
    Training and validation NRMSE obtained using pseudo inverse with number of training samples: 
    Use the same weights and  noise variance, but vary the number of training samples. 
    But first, for each number of training samples, try new random seeds (say 11 to 21) 
    to generate the training data. 
    Have a good number of validation samples. 
    What you are looking for is the mean and variance (spread) of NMRSE.
"""

# Creating dataset
sample_count = 100
training_validation_ratio = .8
featurs = 15
eta= 1
max_iter = 20
min_change_NRMSE  = 99999
lambda2 = 0
lambda1 = 0
nrmse_list=[]
for i in range(10):
    training_count = int(sample_count*training_validation_ratio)
    sample = matrix_generator(sample_count, featurs)
    training_sample = sample[0:training_count]
    validation_sample = sample[training_count:]
    weight_vector = np.array([1]*featurs)
    bias_w_0= np.array([0]*sample_count)
    noise_varience =  np.random.random()*10
    target_sample = column_generator(sample, weight_vector,bias_w_0, noise_varience)
    target_training_sample = target_sample[0:training_count]
    target_validation_sample = target_sample[training_count:]
    updated_weight, updated_nrmse = estimate_weight_gradient_descent(training_sample, target_training_sample, eta, max_iter, min_change_NRMSE, lambda2=0, lambda1=0)
    nrmse_list.append(updated_nrmse)

fig = plt.figure(figsize =(10, 7))
 
plt.boxplot(nrmse_list)

# show plot
plt.show()


# 
# b) Training and validation NRMSE obtained using pseudo inverse with number of variables 
# 

# In[ ]:





# c) Training and validation NRMSE obtained using pseudo inverse with noise variance [2]
# 

# In[ ]:





# d) Training and validation NRMSE obtained using pseudo inverse with w0 [2]
# 

# In[ ]:





# e) Training and validation NRMSE obtained using pseudo inverse with lambda2 [2]
# 

# In[ ]:





# f) Time taken to solve pseudo inverse with number of samples and its breaking point [2]
# 

# In[ ]:





# g) Time taken to solve pseudo inverse with number of variables and its breaking point [2]
# 

# In[ ]:





# h) Training and validation NRMSE obtained using gradient descent with max_iter [2]
# 

# In[ ]:





# i) Training and validation NRMSE obtained using gradient descent with eta [2]
# 

# In[ ]:





# j) Time taken to solve gradient descent with number of samples and its breaking point [2]
# 

# In[ ]:





# k) Time taken to solve gradient descent with number of variables and its breaking point [2]
# 

# In[ ]:





# l) Training and validation NRMSE and number of nearly zero weights obtained using gradient descent with lambda2 [2]
# 

# In[ ]:





# m) Training and validation NRMSE and number of nearly zero weights obtained using gradient descent with lambda1 [2]
# 

# In[ ]:





# n) Experiment (h) but, this time with number of training samples [2]
# 

# In[ ]:





# o) Experiment (h) but, this time with number of variables [2]
# 

# In[ ]:





# ### Testing Block

# In[14]:


ROW = 3
COLUMN = 5
matrix = matrix_generator(ROW,COLUMN)
weight_vector = matrix_generator(COLUMN, 1)
noise_variance = matrix_generator(ROW, 1)
bais_w_0 = matrix_generator(ROW, 1)
target_y = column_generator(matrix, weight_vector, bais_w_0, noise_variance)
estimate_y=calculate_estimate_y(matrix, weight_vector)
mse=mean_square_error(estimate_y, target_y)
l2_norm=compute_l2_norm(weight_vector)
weight, MSE, pridicted_y =estimate_weight(matrix,target_y,l2_norm)
gradient_weight = compute_gradient_of_MSE(matrix, target_y, weight)
cg_l2 = compute_gradient_of_L2_norm(matrix, weight_vector)
cg_l1 = compute_gradient_of_L1_norm(matrix, weight_vector)
single_update_weight(matrix, target_y, weight_vector, 1 , 0, 0)
wgd = estimate_weight_gradient_descent(matrix, target_y, 1, 10, 50, lambda2=0, lambda1=0)


# 14. Write your overall learning points by doing entire assignment.

# This assignment helps me to learn about python and numpy. how to implement linear regression using python. I came across about differnt Python's Library and their

# In[ ]:




