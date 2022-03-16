# Movie Rating

from numpy import dot
import pandas as pd
import matplotlib.pyplot as plt
from pydotplus.graphviz import graph_from_dot_data
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn import tree
from IPython.display import Image
import pydotplus

data = pd.read_csv('Y:\\Intern\\Decibels\\regressive.csv')
# print(data.head()) # Prints data in more convenient form
# print(data.info())  # Gives details about the file with Range of the entries,names of columns. Non-null count indicates columns having none value. 

# In some datas the data type comes as object instead as float,int.
# Also for some non null count, the count is less than others. 
# This inconsistence may cause an error. Hence we use dummy function

data = pd.get_dummies(data,columns=['3D_available','Genre'])
# print(data.info()) # The object data types gets converted into unsigned int 

# Still the column Time taken has less no of elements than others. hence
data['Time_taken']=data['Time_taken'].fillna(data['Time_taken'].mean())
# Here we are filling the the remaining data with the mean time values. Mean is taken to reduce the error

y = data[['Collection']] #Labels
x=data.drop(['Collection'],axis=1) # Removes collection column from data

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25)

model=tree.DecisionTreeRegressor(max_depth=4)
model.fit(xtrain,ytrain)

ypredict = model.predict(xtrain) # will give prediction as values

# However no one can understand the prediction from these values. So this values are to be converted in a tree 
# dot = tree.export_graphviz(model,out_file= None)
# graph = pydotplus.graph_from_dot_data(dot)
# Image(graph.create_png()) # This creates the values into tree format but here the features are not specified properly

# Hence we will do this
dot = tree.export_graphviz(model,out_file= None, feature_names= xtrain.columns, filled=True)
# feature_names will names to the features and the filled=true will the nodes(boxes) different colours while filled=false wont give colours
graph = pydotplus.graph_from_dot_data(dot)
Image(graph.create_png())

