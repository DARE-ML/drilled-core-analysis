import numpy as np 

import matplotlib.pyplot as plt


from numpy import *

from sklearn import datasets 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

import random

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text


from sklearn.neural_network import MLPRegressor

from sklearn.tree import DecisionTreeRegressor 

from sklearn.ensemble import RandomForestClassifier
 
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestRegressor 

from sklearn.metrics import roc_curve, auc

from sklearn.decomposition import PCA

def read_data(run_num, prob):

    normalise = False
    
    if prob == 'classifification': #Source:  Pima-Indian diabetes dataset: https://www.kaggle.com/kumargh/pimaindiansdiabetescsv
        data_in = genfromtxt("pima.csv", delimiter=",")
        data_inputx = data_in[:,0:8] # all features 0, 1, 2, 3, 4, 5, 6, 7 
        data_inputy = data_in[:,-1] # this is target - so that last col is selected from data

    elif prob == 'regression': # energy - regression prob
        data_in = genfromtxt('ENB2012_data.csv', delimiter=",")  # you can replace this with Abalone
        data_inputx = data_in[:,0:8] # all features 0, - 7
        data_inputy = data_in[:,8] # this is target - just the heating load selected from data


  

    if normalise == True:
        transformer = Normalizer().fit(data_inputx)  # fit does nothing.
        data_inputx = transformer.transform(data_inputx)
 

 
    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.40, random_state=run_num)

    return x_train, x_test, y_train, y_test



def dimen_reduction(x_train, x_test, type_model):

    #Scikit-learn is using SVD for PCA

    if type_model ==0: #SVD solver - full
        pca = PCA(n_components=3, svd_solver='full')  # SVD LAPACK Solver
    elif type_model ==1: #SVD 
        pca = PCA(n_components=3, svd_solver='arpack') # SVD AROACK Solver

    #note number of components can be changed to 0.95 (but since we have different train and test data - that can create problems)
    # it is best to combine both train and test data and then split them back again


    #data = np.vstack((x_train,x_test)) # something along these lines

    #print(data.shape, ' * ')


    reduced_datatrain= pca.fit_transform(x_train)
    train_varianceratio = pca.explained_variance_ratio_

    reduced_datatest = pca.fit_transform(x_test)
    test_varianceratio = pca.explained_variance_ratio_



    return reduced_datatrain, reduced_datatest, test_varianceratio, train_varianceratio

 
    
def scipy_models(x_train, x_test, y_train, y_test, type_model, hidden, learn_rate, run_num, problem):
 

 

    if problem == 'classifification':
        if type_model ==0: #SGD 
            model = MLPClassifier(hidden_layer_sizes=(hidden,), random_state=run_num, max_iter=100,solver='sgd',  learning_rate_init=learn_rate )  
        elif type_model ==1: #https://scikit-learn.org/stable/modules/tree.html  (see how tree can be visualised)
            model = DecisionTreeClassifier(random_state=0, max_depth=tree_depth) 
        elif type_model ==2:
            model = RandomForestClassifier(n_estimators=100, max_depth=tree_depth, random_state=run_num)
             

    elif problem == 'regression':
        if type_model ==0: #SGD  
            model = MLPRegressor(hidden_layer_sizes=(hidden*3,), random_state=run_num, max_iter=500, solver='adam',learning_rate_init=learn_rate) 
        elif type_model ==1:  
            model = DecisionTreeRegressor(random_state=0, max_depth=tree_depth)
        elif type_model ==2: 
            model = RandomForestRegressor(n_estimators=100, max_depth=tree_depth, random_state=run_num) 

       
   
    # Train the model using the training sets

    model.fit(x_train, y_train)
   

    if type_model ==1:
        r = export_text(model)
        print(r)


    # Make predictions using the testing set
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train) 

    if problem == 'regression':
        perf_test =  np.sqrt(mean_squared_error(y_test, y_pred_test)) 
        perf_train=  np.sqrt(mean_squared_error(y_train, y_pred_train)) 

    if problem == 'classifification': 
        perf_test = accuracy_score(y_pred_test, y_test) 
        perf_train = accuracy_score(y_pred_train, y_train) 
        cm = confusion_matrix(y_pred_test, y_test) 
        #print(cm, 'is confusion matrix')
        #auc = roc_auc_score(y_pred, y_test, average=None) 

    return perf_test #,perf_train


def main(): 

    max_expruns = 5

    nn_all = np.zeros(max_expruns) 
    nnpca_all = np.zeros(max_expruns)   

    learn_rate = 0.01
    hidden = 8

    prob = 'classifification' #  classification  or regression 
    #prob = 'regression' #  classification  or regression 


    # classifcation accurary is reported for classification and RMSE for regression

    print(prob, ' is our problem')

 
 
    for run_num in range(0,max_expruns): 

        x_train, x_test, y_train, y_test = read_data(run_num, prob)   
        
        acc_nn= scipy_models(x_train, x_test, y_train, y_test, 0, hidden, learn_rate, run_num, prob) #SGD 


        [reduced_datatrain, reduced_datatest, variance_scoretrain, variance_scoretest] =  dimen_reduction(x_train, x_test, 0) # 0 is for PCA (you can try 1 for case of SVD)

        print(reduced_datatrain, variance_scoretrain, ' reduced_datatrain - variance_scoretrain')


        acc_nnpca = scipy_models(reduced_datatrain, reduced_datatest, y_train, y_test, 0, hidden, learn_rate, run_num, prob) #SGD  after PCA
       
        nn_all[run_num] = acc_nn
        nnpca_all[run_num] = acc_nnpca
    print(nn_all,' nn_all')
    print(np.mean(nn_all), ' mean nn_all')
    print(np.std(nn_all), ' std nn_all')
 
    print(nnpca_all,  ' tree_all')
    print(np.mean(nnpca_all),  ' mean - nn pca _all')
    print(np.std(nnpca_all),  ' std - nn - pca _all') 

 


if __name__ == '__main__':
     main() 