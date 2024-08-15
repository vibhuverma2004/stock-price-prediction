#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import necessary library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from torchvision import datasets, models, transforms
import os
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


# In[3]:


pd.pandas.set_option('display.max_columns', None)

url = 'https://www.ee.iitb.ac.in/~asethi/Dump/MouseTrain.csv'
df = pd.read_csv(url)


# In[4]:


sample_count, feature_count = df.shape
plt.rcParams['figure.figsize']=(feature_count, feature_count)
sns.heatmap(df.corr())
plt.show()


# ### 2. Perform exploratory data analysis to find out: [3]
# a. Which variables are usable, and which are not?
# 
# b. Are there significant correlations among variables?
# 
# c. Are the classes balanced?

# In[4]:


# Calculate the correlation matrix
corr_matrix = df.corr().abs()

# Set the threshold
threshold = 0.7

# Identify highly correlated features
high_corr_features = np.where(corr_matrix > threshold)

# Remove highly correlated features
high_corr_features = [(corr_matrix.columns[x], corr_matrix.columns[y]) for x, y in zip(*high_corr_features) if x != y and x < y]
for feature in high_corr_features:
    df.drop(feature[1], axis=1 ,inplace=True ,errors='ignore')
    
# Print the modified dataset
#print(df)


# In[5]:


sample_count, feature_count = df.shape


# In[6]:


plt.rcParams['figure.figsize']=(feature_count,feature_count)
sns.heatmap(df.corr())
plt.show()


# 2 c. To check if the classes are balanced, we can use the value_counts() function to count the number of samples in each class. In this code, we print the result of value_counts() for the 'class' column. From the output, we can see that there are 50 samples in class 0 and 51 samples in class 1, so the classes are nearly balanced.

# In[7]:


# Check the class distribution
print(df['Treatment_Behavior'].value_counts())


# In[8]:



print(df['Genotype'].value_counts())


# 3) Develop a strategy to deal with missing variables. You can choose to impute the variable. The recommended way is to use multivariate feature imputation
# 

# In[9]:


# Load the dataset from the URL
url = 'https://www.ee.iitb.ac.in/~asethi/Dump/MouseTrain.csv'
df = pd.read_csv(url)

# Identify variables with missing values
vars_with_missing = df.columns[df.isnull().any()].tolist()

# Choose an imputation method
imputer = KNNImputer(n_neighbors=5)

# Impute missing values
df[vars_with_missing] = imputer.fit_transform(df[vars_with_missing])

# Evaluate the imputation results
# print("Missing values after imputation:\n", df.isnull().sum())


# 4) Select metrics that you will use, such as accuracy, F1 score, balanced accuracy, AUC etc. Remember, youhave two separate classification tasks – one is binary, the other has four classes. You may have to do somereading about multi-class classification metrics.

# For the binary classification task, the following metrics could be used:
# 
# Accuracy: It measures the percentage of correctly classified samples out of the total number of samples.
# 
# Precision: It is the ratio of true positives to the total number of predicted positives. It measures how many of the predicted positive samples were actually positive.
# 
# Recall: It is the ratio of true positives to the total number of actual positives. It measures how many of the actual positive samples were correctly identified as positive.
# 
# F1 score: It is the harmonic mean of precision and recall. It balances both precision and recall and provides a single metric to evaluate the performance of a binary classifier.
# 
# AUC-ROC: It is the area under the receiver operating characteristic curve. It measures the ability of the binary classifier to distinguish between positive and negative samples.
# 
# For the multi-class classification task with four classes, the following metrics could be used:
# 
# Accuracy: It measures the percentage of correctly classified samples out of the total number of samples.
# 
# Precision: It is the ratio of true positives to the total number of predicted positives for each class. It measures how many of the predicted positive samples for each class were actually positive.
# 
# Recall: It is the ratio of true positives to the total number of actual positives for each class. It measures how many of the actual positive samples for each class were correctly identified as positive.
# 
# F1 score: It is the harmonic mean of precision and recall for each class. It balances both precision and recall for each class and provides a single metric to evaluate the performance of a multi-class classifier.
# 
# Balanced accuracy: It is the average of recall for each class. It takes into account the imbalance in class distribution and provides a single metric to evaluate the overall performance of a multi-class classifier.
# 
# 
# The choice of the best metric depends on the specific context and goals of the classification task. However, in general, the F1 score is a commonly used and reliable metric for evaluating the performance of a classifier, particularly in binary classification tasks where the classes are balanced. For imbalanced datasets or multi-class classification tasks, the balanced accuracy or F1 score weighted by class frequency could be a better choice.
# 
# However, it's important to note that no single metric can provide a complete picture of the performance of a classifier, and it's often necessary to consider multiple metrics in combination to make an informed decision. Therefore, it's recommended to evaluate the performance of a classifier using a combination of metrics such as F1 score, accuracy, and AUC-ROC.
# 
# 
# 
# 
# 
# 

# 5. Using five-fold cross-validation (you can use GridSearchCV from scikit-learn) to find the reasonable (I cannotsay “best” because you have two separate classifications to perform) hyper-parameter settings for thefollowing model types:
# 
# a. Linear SVM with regularization as hyperparameter 
# 
# b. RBF kernel SVM with kernel width and regularization as hyperparameters 
# 
# c. Neural network with single ReLU hidden layer and Softmax output (hyperparameters: number ofneurons, weight decay) [2]
# 
# d. Random forest (max tree depth, max number of variables per node) 

# a. Linear SVM with regularization as hyperparameter

# In[11]:


# Define a linear SVM model
svm = SVC(kernel="linear")

# Define the hyperparameters to tune over
svm_params = {"C": [0.01, 0.1, 1, 10, 100]}

# Define the grid search object, using 5-fold cross-validation and accuracy as the scoring metric
svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring="accuracy")

# Fit the grid search object to the data, using all features except for "Genotype" and "Treatment_Behavior" as input and "Genotype" as the target
svm_grid.fit(df.drop(["Genotype", "Treatment_Behavior"], axis=1), df["Genotype"])

# Print the best hyperparameters found by the grid search
print("Best hyperparameters for Linear SVM:", svm_grid.best_params_)

# Save the SVM grid search object as a file named "svm_grid"

torch.save(svm_grid , "svm_grid")


# b. RBF kernel SVM with kernel width and regularization as hyperparameters 

# In[12]:




# Define an SVM classifier with RBF kernel
rbf_svm = SVC(kernel="rbf")

# Define a dictionary of hyperparameters to search over
rbf_svm_params = {"C": [0.01, 0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10, 100]}

# Define a grid search object with 5-fold cross-validation and accuracy as the evaluation metric
rbf_svm_grid = GridSearchCV(rbf_svm, rbf_svm_params, cv=5, scoring="accuracy")

# Fit the grid search object to the input data and target labels
rbf_svm_grid.fit(df.drop(["Genotype", "Treatment_Behavior"], axis=1), df["Genotype"])

# Print the best hyperparameters found by the grid search
print("Best hyperparameters for RBF SVM:", rbf_svm_grid.best_params_)

# Save the trained SVM classifier object and selected hyperparameters to a file using PyTorch's `torch.save()` function
torch.save(rbf_svm_grid, 'rbf_svm_grid')


# c. Neural network with single ReLU hidden layer and Softmax output (hyperparameters: number of neurons, weight decay)

# In[14]:


df.drop(["Genotype", "Treatment_Behavior"], axis=1)


# In[13]:



# define pipeline
nn_pipe = make_pipeline(StandardScaler(), 
                         MLPClassifier(activation='relu', solver='adam', max_iter=1000, random_state=42))

 # define hyperparameters to tune
nn_param_grid = {'mlpclassifier__hidden_layer_sizes': [(50,), (100,), (200,)],
                  'mlpclassifier__alpha': [0.0001, 0.001, 0.01]}

#perform grid search with 5-fold cross validation
nn_grid = GridSearchCV(nn_pipe, nn_param_grid, cv=5, scoring='accuracy')
nn_grid.fit(df.drop(["Genotype", "Treatment_Behavior"], axis=1), df["Genotype"])

#print best hyperparameters

print("Best hyperparameters for Neural Network: {}".format(nn_grid.best_params_))

torch.save(nn_grid, 'nn_grid')


# d. Random forest (max tree depth, max number of variables per node)

# In[14]:


# Define a pipeline object that includes a random forest classifier with default parameters
rf_pipe = Pipeline([('clf', RandomForestClassifier(random_state=42))])

# Define a dictionary of hyperparameters to search over
rf_grid = {'clf__max_depth': [None, 10, 20, 30, 40], 'clf__max_features': ['sqrt', 'log2']}

# Define a grid search object with 5-fold cross-validation
rf_cv = GridSearchCV(rf_pipe, rf_grid, cv=5)

# Fit the grid search object to the input data and target labels
rf_cv.fit(df.drop(["Genotype", "Treatment_Behavior"], axis=1), df["Genotype"])

# Print the best hyperparameters and corresponding accuracy found by the grid search
print("Random Forest - Best params:", rf_cv.best_params_, "Accuracy:", rf_cv.best_score_)

# Save the trained random forest classifier object and selected hyperparameters to a file using PyTorch's `torch.save()` function
torch.save(rf_cv, 'rf_cv')


# 6. Check feature importance for each model to see if the same proteins are important for each model. Read upon how to find feature importance

# In[78]:


# Get the best linear SVM model from the grid search
best_svm = svm_grid.best_estimator_

# Fit the model on the training data
X_train, X_test, y_train, y_test = train_test_split(df.drop(["Genotype", "Treatment_Behavior"], axis=1), df["Genotype"], test_size=0.2, random_state=42)
best_svm.fit(X_train, y_train)

# Get the feature importance
feature_importance = np.abs(best_svm.coef_[0])
feature_importance /= np.sum(feature_importance)

# Print the feature importance
for i, importance in enumerate(feature_importance):
    print(f"Feature {i+1}: {importance:.4f}")


# In[19]:


from sklearn.svm import LinearSVC

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(["Genotype", "Treatment_Behavior"], axis=1), df["Genotype"], test_size=0.2, random_state=42
)

# Create a pipeline with MinMaxScaler and LinearSVC
svm_pipeline = make_pipeline(MinMaxScaler(), LinearSVC())

# Fit the pipeline on the training data
svm_pipeline.fit(X_train, y_train)

# Get the feature importance
svm_weights = svm_pipeline.named_steps["linearsvc"].coef_[0]
svm_feature_importance = np.abs(svm_weights) / np.sum(np.abs(svm_weights))

# Print the feature importance
print("LinearSVC Feature Importance:")
for i, importance in enumerate(svm_feature_importance):
    print(f"Feature {i+1}: {importance:.4f}")


# In[20]:


# Split the input data and target labels into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(["Genotype", "Treatment_Behavior"], axis=1), df["Genotype"], test_size=0.2, random_state=42)

# Fit the grid search object to the training data
nn_grid.fit(X_train, y_train)

# Extract the best MLPClassifier model from the grid search
best_mlp = nn_grid.best_estimator_.named_steps['mlpclassifier']

# Get the feature importance for the MLPClassifier
mlp_weights = best_mlp.coefs_[0]
mlp_feature_importance = np.abs(mlp_weights) / np.sum(np.abs(mlp_weights), axis=0)

# Print the feature importance for the MLPClassifier
print("Feature importance for MLPClassifier:")
for i, importance in enumerate(mlp_feature_importance[0]):
    print(f"Feature {i+1}: {importance:.4f}")


# In[21]:


# Get the feature importances for the best random forest classifier found by the grid search
importances = rf_cv.best_estimator_['clf'].feature_importances_

# Get the feature names
features = df.drop(["Genotype", "Treatment_Behavior"], axis=1).columns

# Sort the feature importances in descending order
sorted_importances, sorted_features = zip(*sorted(zip(importances, features), reverse=True))

# Print the feature importances for the best random forest classifier found by the grid search
print("Random Forest - Feature Importance:")
for feature, importance in zip(sorted_features, sorted_importances):
    print(f"{feature}: {importance:.4f}")

 


# 7. See if removing some features systematically will improve your models (e.g. using recursive feature elimination https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html). 

# In[23]:


# Load the data into a Pandas dataframe
data = df
# Separate the target variable from the features
X = data.drop(labels=["Genotype", "Treatment_Behavior"], axis=1)

y = datay = data[["Genotype", "Treatment_Behavior"]]


# Create a Random Forest classifier
rf = RandomForestClassifier()

# Use RFE to select the most important features
rfe = RFE(estimator=rf, n_features_to_select=5, step=1)
X_rfe = rfe.fit_transform(X, y)

# Print the selected feature names
print(X.columns[rfe.support_])


# 8. Finally, test a few promising models on the test data:
# https://www.ee.iitb.ac.in/~asethi/Dump/MouseTest.csv

# In[24]:


print(X.columns[rfe.support_])


# In[25]:



url_2 = 'https://www.ee.iitb.ac.in/~asethi/Dump/MouseTest.csv'
df_2 = pd.read_csv(url_2)

# Identify variables with missing values
vars_with_missing = df_2.columns[df_2.isnull().any()].tolist()

# Choose an imputation method
imputer = KNNImputer(n_neighbors=5)

# Impute missing values
df_2[vars_with_missing] = imputer.fit_transform(df_2[vars_with_missing])


# Load the train and test data
train_data = df
test_data = df_2


X_train = train_data.drop(["Genotype", "Treatment_Behavior"], axis=1)
y_train = train_data[["Genotype", "Treatment_Behavior"]]
y_train = y_train.iloc[:, 0] # extract the first column as a Series
# Train a Random Forest classifier on the train data
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)


# Train an SVM classifier on the train data
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

# Separate the target variable from the features for the test data
X_test = test_data.drop(["Genotype", "Treatment_Behavior"], axis=1)
y_test = test_data[["Genotype", "Treatment_Behavior"]]


X_train = train_data.drop(["Genotype", "Treatment_Behavior"], axis=1)
y_train = train_data[["Genotype", "Treatment_Behavior"]]



# Make predictions on the test data using the Random Forest classifier
rf_predictions = rf_classifier.predict(X_test)
# Extract the Genotype column from y_test
y_test_genotype = y_test["Genotype"]
rf_accuracy = accuracy_score(y_test_genotype, rf_predictions)
print("Random Forest accuracy:", rf_accuracy)

# Split the y_test variable into two separate variables
y_test_genotype = y_test["Genotype"]
y_test_treatment = y_test["Treatment_Behavior"]

y_train_genotype = y_train["Genotype"]
y_train_treatment = y_train["Treatment_Behavior"]

# Create separate SVM classifiers for each variable
svm_genotype_classifier = SVC(kernel='linear')
svm_treatment_classifier = SVC(kernel='linear')

# Fit the classifiers on the training data
svm_genotype_classifier.fit(X_train, y_train_genotype)
svm_treatment_classifier.fit(X_train, y_train_treatment)

# Make predictions on the test data using the classifiers
svm_genotype_predictions = svm_genotype_classifier.predict(X_test)
svm_treatment_predictions = svm_treatment_classifier.predict(X_test)

# Calculate the accuracy scores for each variable
svm_genotype_accuracy = accuracy_score(y_test_genotype, svm_genotype_predictions)
svm_treatment_accuracy = accuracy_score(y_test_treatment, svm_treatment_predictions)

print("SVM Genotype accuracy:", svm_genotype_accuracy)
print("SVM Treatment_Behavior accuracy:", svm_treatment_accuracy)


# Objective 2: Practice using pre-trained neural networks to extract domain-specific features for new tasks.
# 
# 9. Read the pytorch tutorial to use a pre-trained “ConvNet as fixed feature extractor” from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html and you can ignore “finetuning theConvNet”. Test this code out to see if it runs properly in your environment after eliminating code blocks that you do not need. 

# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[22]:


# Define data augmentation and normalization transformations for training and validation data
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # randomly crop and resize images to 224x224 pixels
        transforms.RandomHorizontalFlip(),  # randomly flip images horizontally
        transforms.ToTensor(),  # convert images to PyTorch tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize images
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),  # resize images to 256x256 pixels
        transforms.CenterCrop(224),  # center-crop images to 224x224 pixels
        transforms.ToTensor(),  # convert images to PyTorch tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize images
    ]),
}

# Set the path to the image data directory
data_dir = 'data\\hymenoptera_data'

# Create PyTorch ImageFolder datasets for the training and validation sets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}

# Create PyTorch data loaders for the training and validation sets
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

# Get the number of images in the training and validation sets
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# Get the class names (e.g. ant, bee)
class_names = image_datasets['train'].classes

# Set the device to use for training (GPU if available, CPU otherwise)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get the dataset of images for the training set (this line is not needed and can be removed)
imag_dataset = dataloaders['train']


# In[23]:


image_datasets['train']


# 10. Write a function that outputs ResNet18 features for a given input image. Extract features for training images
# (in image_datasets['train']). You should get an Nx512 dimensional array.

# In[72]:


def get_images_features(image_datasets):
    # Define a data loader for the given image datasets
    loader = torch.utils.data.DataLoader(image_datasets,
                                         batch_size=4, shuffle=True, num_workers=8)
    
    # Load a pre-trained ResNet-18 model
    model_ft = models.resnet18(pretrained=True)
    
    # Set the model to training mode
    model_ft.train()
    
    # Get the number of input features for the fully connected layer
    num_ftrs = model_ft.fc.in_features
    
    # Replace the fully connected layer with a new one that outputs 512 features
    model_ft.fc = nn.Linear(num_ftrs, 512)
    
    # Initialize outputs and labels tensors
    outputs = None
    y =  None
    
    # Iterate over the batches of data
    for inputs , labels in loader :
        # Enable gradient calculation
        with torch.set_grad_enabled(True):
            # Forward pass through the ResNet-18 model
            x = model_ft(inputs)
            
            # Concatenate the outputs and labels across batches
            if outputs is None :
                outputs = x
                y = labels 
            else :
                outputs = torch.cat((x, outputs), dim=0)
                y = torch.cat((y, labels), dim = 0)
    
    # Return the concatenated outputs and labels
    return outputs, y  


# In[73]:


x_train , y_train = get_images_features(image_datasets['train'])


# In[74]:


x_val , y_val = get_images_features(image_datasets['val'])


# In[75]:


x_test = x_train.detach().numpy()
y_test = y_train.detach().numpy()
x_valid = x_val.detach().numpy()
y_valid = y_val.detach().numpy()


# 11. Compare L2 regularized logistic regression, RBF kernel SVM (do grid search on kernel width and
# regularization), and random forest (do grid search on max depth and number of trees). Test the final model
# on test data and show the results -- accuracy and F1 score.

# In[76]:



# Train a Random Forest classifier on the train data
rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_classifier.fit(x_test, y_test)
rf_predictions = rf_classifier.predict(x_valid)

# Make predictions on the test data using the Random Forest classifier
rf_accuracy = accuracy_score(y_valid, rf_predictions)
print("Random Forest accuracy:", rf_accuracy)
f1 = f1_score(y_valid, rf_predictions, average='weighted')
print('F1 score RF:', f1)

# Train an SVM classifier on the train data
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(x_test, y_test)

# Make predictions on the test data using the classifiers
svm_prediction = svm_classifier.predict(x_valid)

# Calculate the accuracy scores for each variable
svm_accuracy = accuracy_score(y_valid, svm_prediction)

print("SVM accuracy:", svm_accuracy)

f1 = f1_score(y_valid, svm_prediction, average='weighted')
print('F1 score SVM:', f1)


# In[77]:


clf = LogisticRegression(penalty='l2', C=0.01, solver='lbfgs', max_iter=1000)
clf.fit(x_test, y_test)
y_pred = clf.predict(x_valid)
accuracy = accuracy_score(y_valid, y_pred)
print('Accuracy:', accuracy)


f1 = f1_score(y_valid, y_pred, average='weighted')
print('F1 score:', f1)


# 12. Summarize your findings and write your references. 

# for question number 2, I copied " sns.heatmap(df.corr())
# plt.show() "  from https://seaborn.pydata.org/generated/seaborn.heatmap.html
# 
# I refered
# https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on

# for question no.3 I refered https://scikit-learn.org/stable/modules/impute.html

# for question no.5 I refered https://scikit-learn.org/stable/modules/grid_search.html and modefied this portion "param_grid = {'max_depth': [3, 5, 10],
# ...               'min_samples_split': [2, 5, 10]}
# 
# >>> base_estimator = RandomForestClassifier(random_state=0)
# 
# >>> X, y = make_classification(n_samples=1000, random_state=0)
# 
# >>> sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
# 
# ...                          factor=2, resource='n_estimators',
# 
# ...                          max_resources=30).fit(X, y)"
# 
# https://stackoverflow.com/questions/56310965/what-value-is-returned-in-a-grid-search-score
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# for question no.6   https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html

# for question no.7 https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

# for question no.8  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#     
#     https://www.kaggle.com/code/prashant111/svm-classifier-tutorial 

# for question no.9 and 10 https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html 

# for question no.11  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html 

# I discussed problem number 9 and 10 of this assignment with Soutrik Sarangi(20b090014) 
