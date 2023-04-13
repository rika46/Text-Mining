# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:09:33 2023

@author:Chanthrika Palanisamy
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.svm import SVC
import numpy as np
from sklearn.svm import LinearSVC
crypto_final = pd.read_csv(r"/Users/rika/Documents/TM/crypto_final_final.csv")
crypto_final = crypto_final[['LABEL','Headline']]
crypto_final['Headline']=crypto_final['Headline'].values.astype('U')


############## MultiNomial NB for Label Prediction 
vectorizer = CountVectorizer(input="content",lowercase=True,max_features=3000)          # Initiate Count Vectorizer object
content_list = crypto_final['Headline'].tolist() # Convert dataframe column to a list
vec_matrix = vectorizer.fit_transform(content_list)
vec_array = vec_matrix.toarray()
crypto_cv = pd.DataFrame(vec_array, columns=vectorizer.get_feature_names_out())
crypto_cv=crypto_cv.fillna(0)


t_vectorizer = TfidfVectorizer(input="content",lowercase=True, max_features=500000)
t_matrix = t_vectorizer.fit_transform(content_list)
t_array = t_matrix.toarray()
data_tfidf = pd.DataFrame(data=t_array, columns=t_vectorizer.get_feature_names_out())
data_tfidf=data_tfidf.fillna(0)

# Now the dataframe is ready, split into test/train
X_train, X_test, y_train, y_test = train_test_split(crypto_cv,crypto_final['LABEL'],random_state=3763,test_size=0.20)
# Instantiate Multinomial NB
MyModelNB= MultinomialNB()
# Fit the trained model
MyModelNB.fit(X_train, y_train)
# Predict using the fitted model
Prediction = MyModelNB.predict(X_test)
# Build a confusion matrix
cnf_matrix = confusion_matrix(y_test, Prediction)
# Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
cm_df = pd.DataFrame(cnf_matrix,
                     index = ['bitcoin','tether','dogecoin', 'polkadot'], 
                     columns = ['bitcoin','tether','dogecoin', 'polkadot']) 
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

#Plotting the confusion matrix
# https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix ')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

accuracy_score(y_test,Prediction)

# Take 4 pictures of the text and train of X & Y.
X_train, X_test, y_train, y_test = train_test_split(data_tfidf,crypto_final['LABEL'],random_state=3763,test_size=0.20)
# Instantiate Multinomial NB
MyModelNB= MultinomialNB()
# Fit the trained model
MyModelNB.fit(X_train, y_train)
# Predict using the fitted model
Prediction = MyModelNB.predict(X_test)
# Build a confusion matrix
cnf_matrix = confusion_matrix(y_test, Prediction)
# Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
cm_df = pd.DataFrame(cnf_matrix,
                     index = ['bitcoin','tether','dogecoin', 'polkadot'], 
                     columns = ['bitcoin','tether','dogecoin', 'polkadot']) 
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

#Plotting the confusion matrix
# https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

accuracy_score(y_test,Prediction)



##########################################################################
############################ DECISION TREE ###############################
##########################################################################

# Instantiate the decision tree
MyDT=DecisionTreeClassifier(criterion='gini',splitter='best', 
                            max_depth=4, random_state=3232)
# Fit the training data to the model
MyDT.fit(X_train, y_train)
# Plot the tree in a simple way
tree.plot_tree(MyDT)
# Run the test data over this model
DT_pred=MyDT.predict(X_test)
# Build a confusion matrix
bn_matrix = confusion_matrix(y_test, DT_pred)

cmbn_df = pd.DataFrame(bn_matrix,
                     index = ['bitcoin','tether','dogecoin', 'polkadot'], 
                     columns = ['bitcoin','tether','dogecoin', 'polkadot']) 


#Plotting the confusion matrix
# https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
plt.figure(figsize=(5,4))
sns.heatmap(cmbn_df, annot=True)
plt.title('Confusion Matrix - Gini')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()
fig, axes = plt.subplots(figsize=(12, 8))
tree.plot_tree(MyDT, 
               feature_names=X_train.columns,
               class_names=['bitcoin','tether','dogecoin', 'polkadot'],
               filled=True,
               ax=axes)
plt.title('Decision Tree - gini')
plt.show()
# Accuracy
accuracy_score(y_test,DT_pred)


##################### TREE 2 #####################

# Instantiate the decision tree
MyDT=DecisionTreeClassifier(criterion='entropy',splitter='best', 
                            max_depth=4, random_state=3232)
# Fit the training data to the model
MyDT.fit(X_train, y_train)
# Plot the tree in a simple way
tree.plot_tree(MyDT)
# Run the test data over this model
DT_pred=MyDT.predict(X_test)
# Build a confusion matrix
bn_matrix = confusion_matrix(y_test, DT_pred)

cmbn_df = pd.DataFrame(bn_matrix,
                     index = ['bitcoin','tether','dogecoin', 'polkadot'], 
                     columns = ['bitcoin','tether','dogecoin', 'polkadot']) 


#Plotting the confusion matrix
# https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
plt.figure(figsize=(5,4))
sns.heatmap(cmbn_df, annot=True)
plt.title('Confusion Matrix - Entropy')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()
fig, axes = plt.subplots(figsize=(12, 8))
tree.plot_tree(MyDT, 
               feature_names=X_train.columns,
               class_names=['bitcoin','tether','dogecoin', 'polkadot'],
               filled=True,
               ax=axes)
plt.title('Decision Tree - entropy')
plt.show()
# Accuracy
accuracy_score(y_test,DT_pred)

##################### TREE 3 #####################

MyDT=DecisionTreeClassifier(criterion='log_loss',splitter='best', 
                            random_state=341, max_depth=5, min_samples_split=6,
                            min_samples_leaf=20)
# Fit the training data to the model
MyDT.fit(X_train, y_train)
# Plot the tree in a simple way
tree.plot_tree(MyDT)
# Run the test data over this model
DT_pred=MyDT.predict(X_test)
# Build a confusion matrix
bn_matrix = confusion_matrix(y_test, DT_pred)

cmbn_df = pd.DataFrame(bn_matrix,
                     index = ['bitcoin','tether','dogecoin', 'polkadot'], 
                     columns = ['bitcoin','tether','dogecoin', 'polkadot']) 


#Plotting the confusion matrix
# https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
plt.figure(figsize=(5,4))
sns.heatmap(cmbn_df, annot=True)
plt.title('Confusion Matrix - log_loss')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()
fig, axes = plt.subplots(figsize=(12, 8))
tree.plot_tree(MyDT, 
               feature_names=X_train.columns,
               class_names=['bitcoin','tether','dogecoin', 'polkadot'],
               filled=True,
               ax=axes)
plt.title('Decision Tree - log_loss')
plt.show()
# Accuracy
accuracy_score(y_test,DT_pred)



##################### SVM Linear kernel #####################



clf = SVC(C=34, kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
svm1_pred = clf.predict(X_test)

print(accuracy_score(y_test,svm1_pred))

conf_matrix = confusion_matrix(y_test, svm1_pred)
print("\nThe confusion matrix of SVM Linear model is:")
print(conf_matrix)
cmbn_df = pd.DataFrame(bn_matrix,
                     index = ['bitcoin','tether','dogecoin', 'polkadot'], 
                     columns = ['bitcoin','tether','dogecoin', 'polkadot']) 


#Plotting the confusion matrix
# https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
plt.figure(figsize=(5,4))
sns.heatmap(cmbn_df, annot=True)
plt.title('Confusion Matrix: SVM - Linear kernel')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

# Accuracy
accuracy_score(y_test,DT_pred)

##################### SVM polynomial kernel #####################



clf = SVC(C=34, kernel='poly') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
svm2_pred = clf.predict(X_test)

print(accuracy_score(y_test,svm2_pred))

conf_matrix = confusion_matrix(y_test, svm2_pred)
print("\nThe confusion matrix of SVM Linear model is:")
print(conf_matrix)

# Accuracy
accuracy_score(y_test,DT_pred)
cmbn_df = pd.DataFrame(conf_matrix,
                     index = ['bitcoin','tether','dogecoin', 'polkadot'], 
                     columns = ['bitcoin','tether','dogecoin', 'polkadot']) 


#Plotting the confusion matrix
# https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
plt.figure(figsize=(5,4))
sns.heatmap(cmbn_df, annot=True)
plt.title('Confusion Matrix: SVM - Polynomial kernel')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

# Accuracy
accuracy_score(y_test,DT_pred)

##################### SVM Radial kernel #####################



clf = SVC(C=34, kernel='rbf') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
svm3_pred = clf.predict(X_test)

print(accuracy_score(y_test,svm3_pred))

conf_matrix = confusion_matrix(y_test, svm3_pred)
print("\nThe confusion matrix of SVM Linear model is:")
print(conf_matrix)



cmbn_df = pd.DataFrame(bn_matrix,
                     index = ['bitcoin','tether','dogecoin', 'polkadot'], 
                     columns = ['bitcoin','tether','dogecoin', 'polkadot']) 


#Plotting the confusion matrix
# https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
plt.figure(figsize=(5,4))
sns.heatmap(cmbn_df, annot=True)
plt.title('Confusion Matrix: SVM - rbf kernel')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

# Accuracy
accuracy_score(y_test,DT_pred)




from textblob import TextBlob
# function to calculate subjectivity
def getSubjectivity(review):
    return TextBlob(review).sentiment.subjectivity
    # function to calculate polarity
def getPolarity(review):
    return TextBlob(review).sentiment.polarity

# function to analyze the reviews
def analysis(score):
    if score < 0:
        return 'Negative'
    else:
        return 'Positive'
 
crypto_final_senti = crypto_final[['Headline']]
crypto_final_senti['Polarity'] = crypto_final_senti['Headline'].apply(getPolarity) 
crypto_final_senti['Analysis'] = crypto_final_senti['Polarity'].apply(analysis)
crypto_final_senti.head()


crypto_final_senti= crypto_final_senti[['Headline', 'Analysis']]

vectorizer = CountVectorizer(input="content",lowercase=True,max_features=3000)          # Initiate Count Vectorizer object
content_list = crypto_final_senti['Headline'].tolist() # Convert dataframe column to a list
vec_matrix = vectorizer.fit_transform(content_list)
vec_array = vec_matrix.toarray()
crypto_cv_senti = pd.DataFrame(vec_array, columns=vectorizer.get_feature_names_out())
crypto_cv_senti = crypto_cv_senti.fillna(0)

Xs_train, Xs_test, ys_train, ys_test = train_test_split(crypto_cv_senti,crypto_final_senti['Analysis'],random_state=3763,test_size=0.20)


clf = SVC(C=3, kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(Xs_train, ys_train)

#Predict the response for test dataset
svms_pred = clf.predict(Xs_test)

print(accuracy_score(ys_test,svms_pred))

conf_matrix = confusion_matrix(ys_test, svms_pred)
print("\nThe confusion matrix of SVM Linear model is:")
print(conf_matrix)



cmbn_df = pd.DataFrame(conf_matrix,
                     index = ['positive','negative'], 
                     columns = ['positive','negative']) 


#Plotting the confusion matrix
# https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
plt.figure(figsize=(5,4))
sns.heatmap(cmbn_df, annot=True)
plt.title('Confusion Matrix: SVM - sentimental analysis')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


SVM_Model = LinearSVC()
SVM_Model.fit(Xs_train, ys_train)
top_features = 10
coef = SVM_Model.coef_.ravel()
top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
print(coef[top_coefficients])

def plot_coefficients(MODEL=SVM_Model, COLNAMES=X_train.columns, top_features=10):
    ## Model if SVM MUST be SVC, RE: SVM_Model=LinearSVC(C=10)
    coef = MODEL.coef_.ravel()
    top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
    top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
    plt.bar(  x=  np.arange(2 * top_features)  , height=coef[top_coefficients], width=.5,  color=colors)
    feature_names = np.array(COLNAMES)
    plt.xticks(np.arange(0, (2*top_features)), feature_names[top_coefficients], rotation=60, ha="right")
    plt.show()

plot_coefficients()
