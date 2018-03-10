from __future__ import division
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
from scipy import special
import scipy
import matplotlib.pyplot as plt
import time, os, math, warnings

"""
This file contains various functions related to the data-driven prediction in multiphase 
reservoir flow problems. More specifically, given flow rates and pressure only, without knowing 
the physical parameters of the well and reservoir, we want to see how good machine/deep learning
can be in predicting future events.

The scope of this project includes:
- Handcrafted feature formulation for pressure and flow rate problem. 
  Function `buildfeature()`, `build_Ei_feature()`, and `structure_features` are used to play 
  around with the feature set-up.
- Performance of 10 algorithms in solving pressure and flow rate time-series problem. 
  Function `build_pipe()` is used for this purpose; it trains the data using 10 algorithms, 
  performs extensive hyperparameters search, and computes prediction scores.
- How complexities (noise, heterogeneity, anisotropy, etc.) affect the result of the ML prediction (on-going)
"""

def load_data(filename):
    """
    This function reads csv file from the given url and extracts time-series data of oil & water rate, 
    pressure, and water cut.
    :param filename: the url of csv file (stored in GitHub)
    :return: x: matrix of flowrates in all wells and p: bottom hole pressure of observation well
    """
    os.chdir('C:\\Users\\E460\\PycharmProjects\\untitled3\\Research\\csv files')
    df = pd.read_csv(filename)

    t = df.loc[:,['TIME']] # Time in simulation: DAY
    t* = 24 # Converting time from DAY to HOUR
    qo = df.loc[:,['WOPR:P1', 'WOPR:P2', 'WOPR:P3']]
    qw = df.loc[:,['WWPR:P1', 'WWPR:P2', 'WWPR:P3']]
    p = df.loc[:,['WBHP:P1']]
    wc = df.loc[:,['WWCT:P1', 'WWCT:P2', 'WWCT:P3']]
    x = pd.concat([t,qo,qw,wc],axis=1,join='inner')
    return x,p

def buildfeature(tset, qset):
    """
    This function constructs features (including logarithmic and exponential features) 
    from raw flow rate data by capturing the convolution of flow rate change events into a matrix.
    
    :param qset: list of flow rate change events
    :param tset: list of time corresponding to above flow rate change events
    
    :return: matrix of features
    """
    warnings.filterwarnings("ignore")

    k = 50            # Perm in mD
    por = 0.2         # Porosity in fraction
    m = 2             # Viscosity in cP
    re = 1000      # Re in ft
    c = 10**-5       # Total compressibility in 1/psi

    ### Initialization
    # creating time array
    tstep=1
    tset=tset.tolist()
    qset=qset.tolist()
    numdata=int((round(max(tset)-min(tset))/tstep))+1
    t=np.arange(0,round(max(tset)-min(tset))+1,tstep)

    # creating rate array
    q=[]
    for i in range(len(tset)):
        if i==0: # Note: make sure tset starts from 0
            y=np.array([0])
        else:
            y=np.full(int(round((tset[i]-tset[i-1])/tstep)),qset[i-1])
        q=np.concatenate((q,y))


    # Initialize feature array
    f2=np.zeros((numdata,int(len(tset))))
    f3=np.zeros((numdata,int(len(tset))))
    f4=np.zeros((numdata,int(len(tset))))
    f5=np.zeros((numdata,int(len(tset))))
    f6=np.zeros((numdata,int(len(tset))))
    f7=np.zeros((numdata,int(len(tset))))
    f8=np.zeros((numdata,int(len(tset))))
    f9=np.zeros((numdata,int(len(tset))))
    f10=np.zeros((numdata,int(len(tset))))

    # First Feature: q
    # Building 2nd, 3rd, and 4th Features
    for i in range(1, numdata):
        for j in range(i-1):
            if j==0:
                f2[i][j]=(qset[j])*np.log10(t[i])
                f3[i][j]=(qset[j])*(t[i])
                f4[i][j]=(qset[j])/(t[i])
                f5[i][j]=qset[j]/np.exp(t[i])
                f6[i][j]=qset[j]/np.exp(t[i])*np.log10(t[i])
                f7[i][j]=qset[j]/np.exp(2*t[i])*np.log10(t[i])
                f8[i][j]=qset[j]/np.exp(3*t[i])*np.log10(t[i])
                f9[i][j]=qset[j]/(t[i])/np.exp(t[i])
                f10[i][j]=qset[j]*scipy.special.expi(-948*por*m*c*re**2/k/t[i])
            else:
                f2[i][j]=(qset[j]-qset[j-1])*np.log10(t[i]-tset[j])
                if math.isnan(f2[i][j])== True or math.isinf(f2[i][j])== True or (t[i]-tset[j])<10**-6:
                    f2[i][j]=0
                f3[i][j]=(qset[j]-qset[j-1])*(t[i]-tset[j])
                if math.isnan(f3[i][j])== True or math.isinf(f3[i][j])== True or (t[i]-tset[j])<10**-6:
                    f3[i][j]=0
                f4[i][j]=(qset[j]-qset[j-1])/(t[i]-tset[j])
                if math.isnan(f4[i][j])== True or math.isinf(f4[i][j])== True or (t[i]-tset[j])<10**-6:
                    f4[i][j]=0
                #f5[i][j]=(qset[j]-qset[j-1])*scipy.special.expi(-948*por*m*c*re**2/k/(t[i]-tset[j]))
                f5[i][j]=(qset[j]-qset[j-1])/np.exp((t[i]-tset[j]))
                if math.isnan(f5[i][j])== True or math.isinf(f5[i][j])== True or (t[i]-tset[j])<10**-6:
                    f5[i][j]=0
                f6[i][j]=(qset[j]-qset[j-1])/np.exp(t[i]-tset[j])*(np.log10(t[i]-tset[j]))
                if math.isnan(f6[i][j])== True or math.isinf(f6[i][j])== True or (t[i]-tset[j])<10**-6:
                    f6[i][j]=0
                f7[i][j]=(qset[j]-qset[j-1])/np.exp(2*(t[i]-tset[j]))*(np.log10(t[i]-tset[j]))
                if math.isnan(f7[i][j])== True or math.isinf(f7[i][j])== True or (t[i]-tset[j])<10**-6:
                    f7[i][j]=0
                f8[i][j]=(qset[j]-qset[j-1])/np.exp(3*(t[i]-tset[j]))*(np.log10(t[i]-tset[j]))
                if math.isnan(f8[i][j])== True or math.isinf(f8[i][j])== True or (t[i]-tset[j])<10**-6:
                    f8[i][j]=0
                f9[i][j]=(qset[j]-qset[j-1])/((t[i]-tset[j]))/(np.exp(t[i]-tset[j]))
                if math.isnan(f8[i][j])== True or math.isinf(f9[i][j])== True or (t[i]-tset[j])<10**-6:
                    f9[i][j]=0
                f10[i][j]=(qset[j]-qset[j-1])*scipy.special.expi(-948*por*m*c*re**2/k/(t[i]-tset[j]))
                if math.isnan(f10[i][j])== True or math.isinf(f10[i][j])== True or (t[i]-tset[j])<10**-6:
                    f10[i][j]=0
    f2tot = np.sum(f2,axis=1)
    f3tot = np.sum(f3,axis=1)
    f4tot = np.sum(f4,axis=1)
    f5tot = np.sum(f5,axis=1)
    f6tot = np.sum(f6,axis=1)
    f7tot = np.sum(f7,axis=1)
    f8tot = np.sum(f8,axis=1)
    f9tot = np.sum(f9,axis=1)
    f10tot = np.sum(f10,axis=1)

    features=[0,q,f2tot, f3tot, f4tot, f5tot, f6tot, f7tot, f8tot, f9tot, f10tot]

    return features

def build_Ei_feature(tset, qset):
    """
    This function constructs exponential integral features  
    from raw flow rate data by capturing the convolution of flow rate change events into a matrix.
    
    :param qset: list of flow rate change events
    :param tset: list of time corresponding to above flow rate change events
    
    :return: matrix of features
    """
    warnings.filterwarnings("ignore")
    ### Initialization
    # creating time array
    tstep = 1
    tset = tset.tolist()
    qset = qset.tolist()
    numdata=int((round(max(tset)-min(tset))/tstep))+1
    t= np.arange(0,round(max(tset)-min(tset))+1,tstep)

    # creating rate array

    # Initialize feature array
    f1 = np.zeros((numdata,int(len(tset))))
    f2 = np.zeros((numdata,int(len(tset))))
    f3 = np.zeros((numdata,int(len(tset))))
    f4 = np.zeros((numdata,int(len(tset))))
    f5 = np.zeros((numdata,int(len(tset))))
    f6 = np.zeros((numdata,int(len(tset))))
    f7 = np.zeros((numdata,int(len(tset))))
    f8 = np.zeros((numdata,int(len(tset))))

    # First Feature: q
    # Building 2nd, 3rd, and 4th Features
    for i in range(1,numdata):
        for j in range(i-1):
            if j==0:
                f1[i][j] = qset[j]*scipy.special.expi(-10**-85/t[i])
                f2[i][j] = qset[j]*scipy.special.expi(-2*10**-85/t[i])
                f3[i][j] = qset[j]*scipy.special.expi(-3*10**-85/t[i])
                f4[i][j] = qset[j]*scipy.special.expi(-4*10**-85/t[i])
                f5[i][j] = qset[j]*scipy.special.expi(-5*10**-85/t[i])
                f6[i][j] = qset[j]*scipy.special.expi(-6*10**-85/t[i])
                f7[i][j] = qset[j]*scipy.special.expi(-7*10**-85/t[i])
                f8[i][j] = qset[j]*scipy.special.expi(-8*10**-85/t[i])
            else:
                f1[i][j] = (qset[j]-qset[j-1])*scipy.special.expi(-10**-85/(t[i]-tset[j]))
                f2[i][j] = (qset[j]-qset[j-1])*scipy.special.expi(-2*10**-85/(t[i]-tset[j]))
                f3[i][j] = (qset[j]-qset[j-1])*scipy.special.expi(-3*10**-85/(t[i]-tset[j]))
                f4[i][j] = (qset[j]-qset[j-1])*scipy.special.expi(-4*10**-85/(t[i]-tset[j]))
                f5[i][j] = (qset[j]-qset[j-1])*scipy.special.expi(-5*10**-85/(t[i]-tset[j]))
                f6[i][j] = (qset[j]-qset[j-1])*scipy.special.expi(-6*10**-85/(t[i]-tset[j]))
                f7[i][j] = (qset[j]-qset[j-1])*scipy.special.expi(-7*10**-85/(t[i]-tset[j]))
                f8[i][j] = (qset[j]-qset[j-1])*scipy.special.expi(-8*10**-85/(t[i]-tset[j]))


    f1tot = np.sum(f1,axis=1)
    f2tot = np.sum(f2,axis=1)
    f3tot = np.sum(f3,axis=1)
    f4tot = np.sum(f4,axis=1)
    f5tot = np.sum(f5,axis=1)
    f6tot = np.sum(f6,axis=1)
    f7tot = np.sum(f7,axis=1)
    f8tot = np.sum(f8,axis=1)


    features=[0, f1tot, f2tot, f3tot, f4tot, f5tot, f6tot, f7tot, f8tot]

    return features

def structure_features(X_train):
    """
    This function builds multiple features (by calling function 'buildfeatures()') from each flow rate
    and stacks them horizontally into a new feature matrix.
    :param X_train: matrix of raw data
    :return: matrix of constructed features
    """
    t_train = X_train['TIME']

    # Build features
    fa_tr=buildfeature(t_train,X_train['WOPR:P1'])
    fb_tr=buildfeature(t_train,X_train['WOPR:P2'])
    fc_tr=buildfeature(t_train,X_train['WOPR:P3'])
    fa_wr_tr=buildfeature(t_train,X_train['WWPR:P1'])
    fb_wr_tr=buildfeature(t_train,X_train['WWPR:P2'])
    fc_wr_tr=buildfeature(t_train,X_train['WWPR:P3'])

    # Arrange features
    x_train=pd.DataFrame({'fa1':fa_tr[1], 'fa2':fa_tr[2], 'fa3':fa_tr[3], 'fa4':fa_tr[4],'fa6':fa_tr[6],'fa7':fa_tr[7],'fa8':fa_tr[8],
                          'fb1':fb_tr[1], 'fb2':fb_tr[2], 'fb3':fb_tr[3], 'fb4':fb_tr[4],'fb6':fb_tr[6],'fb7':fb_tr[7],'fb8':fb_tr[8],
                          'fc1':fc_tr[1], 'fc2':fc_tr[2], 'fc3':fc_tr[3], 'fc4':fc_tr[4],'fc6':fc_tr[6],'fc7':fc_tr[7],'fc8':fc_tr[8],
                          'faw1':fa_wr_tr[1], 'faw2':fa_wr_tr[2], 'faw3':fa_wr_tr[3], 'faw4':fa_wr_tr[4],'faw6':fa_wr_tr[6],'faw7':fa_wr_tr[7],'faw8':fa_wr_tr[8],
                          'fbw1':fb_wr_tr[1], 'fbw2':fb_wr_tr[2], 'fbw3':fb_wr_tr[3], 'fbw4':fb_wr_tr[4],'fbw6':fb_wr_tr[6],'fbw7':fb_wr_tr[7],'fbw8':fb_wr_tr[8],
                          'fcw1':fc_wr_tr[1], 'fcw2':fc_wr_tr[2], 'fcw3':fc_wr_tr[3], 'fcw4':fc_wr_tr[4],'fcw6':fc_wr_tr[6],'fcw7':fc_wr_tr[7],'fcw8':fc_wr_tr[8]})

    return x_train

def structure_Ei_features(X_train):
    """
    This function builds multiple Ei-function features (by calling function 'build_Ei_features()') 
    from each flow rate and stacks them horizontally into a new feature matrix.
    :param X_train: matrix of raw data
    :return: matrix of constructed features
    """
    t_train=X_train['TIME']

    # Build features
    fa_tr = build_Ei_feature(t_train,X_train['WOPR:P1'])
    fb_tr = build_Ei_feature(t_train,X_train['WOPR:P2'])
    fc_tr = build_Ei_feature(t_train,X_train['WOPR:P3'])
    fa_wr_tr = build_Ei_feature(t_train,X_train['WWPR:P1'])
    fb_wr_tr = build_Ei_feature(t_train,X_train['WWPR:P2'])
    fc_wr_tr = build_Ei_feature(t_train,X_train['WWPR:P3'])

    # Arrange features
    x_train=pd.DataFrame({'fa1':fa_tr[1], 'fa2':fa_tr[2], 'fa3':fa_tr[3], 'fa4':fa_tr[4], 'fa5':fa_tr[5], 'fa6':fa_tr[6],'fa7':fa_tr[7], 'fa8':fa_tr[8],
                          'fb1':fb_tr[1], 'fb2':fb_tr[2],'fb3':fb_tr[3], 'fb4':fb_tr[4], 'fb5':fb_tr[5], 'fb6':fb_tr[6], 'fb7':fb_tr[7], 'fb8':fb_tr[8],
                          'fc1':fc_tr[1], 'fc2':fc_tr[2],'fc3':fc_tr[3], 'fc4':fc_tr[4], 'fc5':fc_tr[5], 'fc6':fc_tr[6], 'fc7':fc_tr[7], 'fc8':fc_tr[8],
                          'faw1':fa_wr_tr[1], 'faw2':fa_wr_tr[2],'faw3':fa_wr_tr[3], 'faw4':fa_wr_tr[4], 'faw5':fa_wr_tr[5], 'faw6':fa_wr_tr[6],'faw7':fa_wr_tr[7], 'faw8':fa_wr_tr[8],
                          'fbw1':fb_wr_tr[1], 'fbw2':fb_wr_tr[2],'fbw3':fb_wr_tr[3], 'fbw4':fb_wr_tr[4], 'fbw5':fb_wr_tr[5], 'fbw6':fb_wr_tr[6], 'fbw7':fb_wr_tr[7], 'fbw8':fb_wr_tr[8],
                          'fcw1':fc_wr_tr[1], 'fcw2':fc_wr_tr[2],'fcw3':fc_wr_tr[3], 'fcw4':fc_wr_tr[4], 'fcw5':fc_wr_tr[5], 'fcw6':fc_wr_tr[6], 'fcw7':fc_wr_tr[7], 'fcw8':fc_wr_tr[8]})

    return x_train

def regression(x_train, y_train):
    """This function trains the data, performs hyperparameters search, and chooses the classifier
    that gives the best cross-val score. This function returns the best performing classifier"""

    clf=GridSearchCV(MLPRegressor(),
                     [{'clf__hidden_layer_sizes':[(10,),(50,),(100,),(500,)],
                       'clf__solver':['lbfgs']}],
                     cv=3)
    clf = GridSearchCV(Ridge(),[{'alpha':[0.01,0.1,1,10,100,1000]}], cv=3)

    clf.fit(x_train, y_train)
    clf=clf.best_estimator_

    return clf

def build_pipe(X_train,y_train,X_test,y_test):
    """
    This function tests 10 different algorithms, performs greedy hyperparameters search
    on each of them, and stores the scores and classifiers into files.
    Unfortunately, some functions require files that are not published online. 
    
    :param X_train: matrix of features in training set
    :param y_train: vector of label in training set
    :param X_test: matrix of features in test set
    :param y_test: vector of label in test set
    """
    x_train = structure_features(X_train)
    x_test = structure_features(X_test)
    cval_list = []
    clf_list = []
    train_score = []
    test_score = []
    scenario = '14 features'
    pipe_dict = {0: 'Ridge Regression',
                 1: 'Kernel Ridge Regression',
                 2: 'SGD Regression',
                 3: 'Support Vector Regression',
                 4: 'Decision Tree',
                 5: 'Random Forest',
                 6: 'AdaBoost',
                 7: 'Gradient Boosting',
                 8: 'k-nearest Neighbors',
                 9: 'Neural Network'}
    classifiers = [Ridge(),
                 KernelRidge(),
                 SGDRegressor(),
                 SVR(kernel='linear'),
                 DecisionTreeRegressor(),
                 RandomForestRegressor(),
                 AdaBoostRegressor(),
                 GradientBoostingRegressor(),
                 KNeighborsRegressor(),
                 MLPRegressor(solver='lbfgs',activation='identity')]
    param_grid = [{'clf__alpha':[0.01,0.1,1,2,5,10,100,1000]},
                [{'clf__alpha':[0.01,0.1,1,2,5,10,100,1000],'clf__kernel':['linear']},{'clf__alpha':[0.1,0.1,1,10,100,1000],'clf__kernel':['polynomial'],'clf__degree':[2,3,5]}],
                {'clf__alpha':[0.001,0.01,0.1,1,2,5,10,100,1000]},
                [{'clf__C':[10000]}],
                {'clf__max_depth': [None, 5, 10, 30, 50]},
                [{'clf__n_estimators':[2, 3, 5, 10, 20],'clf__max_depth': [None, 5, 10, 20, 30, 50]}],
                {'clf__n_estimators':[2, 5, 10, 20, 50, 100]},
                [{'clf__n_estimators':[2, 5, 10, 20, 50, 100],'clf__max_depth': [None, 5, 10, 20, 30, 50]}],
                {'clf__n_neighbors':[2, 3, 5, 10, 20]},
                {'clf__hidden_layer_sizes':[(10,),(20,),(50,),(100,),(1000,),
                                            (20,5),(20,20),(20,5,10),(20,5,20),(50,10,20),
                                            (100,20,50),(1000,50,100),(1000,200,1000),(1000,50,400,600)]}]

    for i in range(len(classifiers)):
        pipe = Pipeline([('scl', StandardScaler()),
                       ('clf',classifiers[i])])
        clf = GridSearchCV(pipe,cv=4,param_grid=param_grid[i])
        clf.fit(x_train, y_train)
        clf_list.append(clf.best_estimator_)
        cval_list.append(clf.best_score_)

        # Save a classifier as file
        joblib.dump(clf.best_estimator_,
                    'C:\\Users\\E460\\PycharmProjects\\untitled3\\Research\\results\\Far LowWC\\classifiers\\'+scenario+'\\'+pipe_dict[i]+'.pkl')

        # Print scores
        train_score.append(clf.score(x_train, y_train))
        test_score.append(clf.score(x_test, y_test))
        print('%s training set accuracy: %.3f' % (pipe_dict[i], train_score[i]))
        print('%s dev set accuracy: %.3f' % (pipe_dict[i], cval_list[i]))
        print('%s test set accuracy: %.3f \n' % (pipe_dict[i], test_score[i]))


    # Identify the most accurate model on test data
    best_acc = 0.0
    best_clf = 0
    best_pipe = ''
    for idx, val in enumerate(clf_list):
        if val.score(x_test, y_test) > best_acc:
            best_acc = val.score(x_test, y_test)
            best_pipe = val
            best_clf = idx
    print('Classifier with best accuracy: %s' % pipe_dict[best_clf])

    # Saving clf details
    dict_clf = pd.DataFrame.from_dict(OrderedDict([('Algorithms',[str(i) for i in clf_list])]))
    writer = pd.ExcelWriter('C:\\Users\\E460\\PycharmProjects\\untitled3\\Research\\results\\Far LowWC\\summary_clf_temp_'+scenario+'.xlsx', engine='xlsxwriter')
    dict_clf.to_excel(writer, index=False)
    writer.save()

    # Constructing DataFrame output
    dict_sum = OrderedDict([('Algorithms',[i for i in pipe_dict.values()]),
                          ('Training Set Accuracy',train_score),
                          ('Dev Set Accuracy',cval_list),
                          ('Test Set Accuracy',test_score)])
    df = pd.DataFrame.from_dict(dict_sum)
    bar_chart(df,'C:\\Users\\E460\\PycharmProjects\\untitled3\\Research\\results\\Far LowWC\\'+scenario+'.png')

    writer = pd.ExcelWriter('C:\\Users\\E460\\PycharmProjects\\untitled3\\Research\\results\\Far LowWC\\summary_report_temp_'+scenario+'.xlsx', engine='xlsxwriter')
    df.to_excel(writer, index=False)
    writer.save()

    # Plot and print p, q, and der in train and dev set
    for i in range(len(clf_list)):
        x_train_train,x_dev,y_train_train,y_dev=train_test_split(x_train,y_train,test_size=0.25,shuffle=False)
        y_pred_train=clf_list[i].predict(x_train_train)
        y_pred_dev=clf_list[i].predict(x_dev)
        y_pred_test=clf_list[i].predict(x_test)

        # Plot and print Training Data (P and Q)
        plot_pressure_rates(X_train,pd.concat([y_train_train,y_dev],axis=0,join='inner'),
                            np.concatenate((y_pred_train,y_pred_dev),axis=0),
                            labelname='Training Data')
        plt.savefig('C:\\Users\\E460\\PycharmProjects\\untitled3\\Research\\results\\Far LowWC\\P and Q\\'+scenario+'\\'+pipe_dict[i]+'_train.png',
                    bbox_inches="tight")

        # Plot and print Test Data (P and Q)
        plot_pressure_rates(X_test,y_test.loc[:,['WBHP:P1']],y_pred_test,labelname='Test Data')
        plt.savefig('C:\\Users\\E460\\PycharmProjects\\untitled3\\Research\\results\\Far LowWC\\P and Q\\'+scenario+'\\'+pipe_dict[i]+'_test.png',
                    bbox_inches="tight")

        # Plot and print derivatives
        derivatives2(clf_list[i],X_test,x_test,y_test)
        plt.savefig('C:\\Users\\E460\\PycharmProjects\\untitled3\\Research\\results\\Far LowWC\\derivatives\\'+scenario+'\\'+pipe_dict[i]+'.png',
                    bbox_inches="tight")

    return df

def bar_chart(df, filename):
    """This function obtains scores from a csv file and visualizes them in a bar chart"""
    n_groups = df.shape[0]
    index_df = df.loc[:,['Algorithms']]
    index_name = [index_df.values[i][0] for i in range(index_df.shape[0])]
    train_acc = df.loc[:,['Training Set Accuracy']]
    dev_acc = df.loc[:,['Dev Set Accuracy']]
    test_acc = df.loc[:,['Test Set Accuracy']]
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8
    rects1 = plt.barh(index, train_acc.values, bar_width,
                     alpha=opacity,
                     color='r',
                     label='Training Acc')
    rects2 = plt.barh(index + bar_width, dev_acc.values, bar_width,
                     alpha=opacity,
                     color='orange',
                     label='Dev Acc')

    rects3 = plt.barh(index + bar_width*2, test_acc.values, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Test Acc')
    plt.xlabel('Accuracy')
    plt.ylabel('Methods')
    plt.title('Accuracy Comparison')
    plt.xlim([0,1])
    plt.yticks(index + bar_width, index_name)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.105),
               fancybox=True, shadow=True, ncol=5,fontsize=9)
    plt.tight_layout()
    plt.show()
    plt.savefig(filename,bbox_inches="tight")

def derivatives(clf):
    """Calculates pressure derivatives given a classifier"""
    df = pd.read_csv('C:\\Users\\E460\\PycharmProjects\\untitled3\\Research\\csv files\\derivatives.csv')
    t = df.loc[:,['TIME']].as_matrix() # Time in simulation: DAY
    t* = 24 # Converting time from DAY to HOUR
    x_test_der=df.loc[:,['fa1', 'fa2', 'fa3','fa4','fa10','faw1', 'faw2', 'faw3','faw4','faw10',
                         'fb1', 'fb2', 'fb3','fb4', 'fb10','fbw1', 'fbw2', 'fbw3','fbw4', 'fbw10',
                         'fc1', 'fc2', 'fc3','fc4', 'fc10','fcw1', 'fcw2', 'fcw3','fcw4', 'fcw10']]

    p_pred=clf.predict(x_test_der)

    # Delta Pressure
    p_act = df.loc[:,['WBHP:P1']].as_matrix()
    dp_act = abs(p_act[0]-p_act)
    dp_pred = abs(p_act[0]-p_pred)

    # Derivatives
    p_der_act = np.zeros(len(p_act)-2)
    p_der_pred = np.zeros(len(p_pred)-2)
    for i in range(1,len(p_act)-1):
        p_der_act[i-1]=t[i]/(t[i+1]-t[i-1])*abs(dp_act[i+1]-dp_act[i-1])
        p_der_pred[i-1]=t[i]/(t[i+1]-t[i-1])*abs(dp_pred[i+1]-dp_pred[i-1])
    plot_derivatives(t,dp_act,dp_pred,p_der_act,p_der_pred)


    # Plot Test Data (P and Q)
    plot_pressure_rates(df,df.loc[:,['WBHP:P1']],p_pred,labelname='Testing')

def derivatives2(clf, X_test, x_test, y_test):
    """This function calculates pressure derivatives given a classifier"""
    t=X_test['TIME']
    p_pred=clf.predict(x_test)

    # Delta Pressure
    p_act=y_test.as_matrix()
    dp_act=abs(p_act[0]-p_act)
    dp_pred=abs(p_pred[0]-p_pred)

    # Derivatives
    p_der_act=np.zeros(len(p_act)-2)
    p_der_pred=np.zeros(len(p_pred)-2)
    for i in range(1,len(p_act)-1):
        p_der_act[i-1]=t[i]/(t[i+1]-t[i-1])*abs(dp_act[i+1]-dp_act[i-1])
        p_der_pred[i-1]=t[i]/(t[i+1]-t[i-1])*abs(dp_pred[i+1]-dp_pred[i-1])
    plot_derivatives(t,dp_act,dp_pred,p_der_act,p_der_pred)

def plot_derivatives(t,dp_act,dp_pred,p_der_act,p_der_pred):
    """This function plots delta pressure and the derivative & shows comparison between actual data and
    prediction result"""
    plt.figure()
    plt.loglog(t, dp_act, 'k-',linewidth=3,label='Actual dP')
    plt.loglog(t, dp_pred, 'ro',label='Predicted dP')
    plt.loglog(t[1:-1], p_der_act, 'k*',linewidth=3,label='Actual Derivative')
    plt.loglog(t[1:-1], p_der_pred, 'gx',label='Predicted Derivative')
    plt.ylim(1,5000)
    plt.xlabel('dt (hours)')
    plt.ylabel('dP & Derivative')
    plt.legend(loc="best", prop=dict(size=12))
    plt.grid()


def plot_pressure(t, p_actual, p_pred, title, color):
    """This function plots actual and predicted bottom hole pressure"""
    # Plotting pwf v time
    plt.plot(t, p_actual, 'k-',linewidth=3,label='Actual Pwf')

    if title == 'Training Data':
        plt.plot(t[0:int(0.7*p_pred.shape[0])], p_pred[0:int(0.7*p_pred.shape[0])], 'rx',markeredgecolor=color,label=title)
        plt.plot(t[int(0.7*p_pred.shape[0]):], p_pred[int(0.7*p_pred.shape[0]):], 'yx',markeredgecolor='orange',label='Dev Set')
    else:
        plt.plot(t, p_pred, 'gx',markeredgecolor=color,label=title)
    plt.xlabel("Time (hours)")
    plt.ylabel("BH Pressure (psi)", fontsize=9)
    plt.title("BH Pressure Well A",y=1, fontsize=9)
    plt.legend(loc="best", prop=dict(size=8))
    plt.xlim(0,max(t))
    plt.ylim(0, max(max(p_actual.values),max(p_pred)))
    plt.grid(True)

def plot_rates(t,q,wellname,color):
    """This function plots actual flow rates"""
    # Plotting Flow Rate v time
    plt.plot(t, q,color, linewidth=3)
    plt.xlabel("Time (hours)")
    plt.ylabel("Flow Rate (STB/D)", fontsize=9)
    plt.title('Flow Rate Well '+wellname,y=0.82, fontsize=9)
    plt.xlim(0,max(t))
    plt.ylim(0,max(q)+10)
    plt.grid(True)

def plot_pressure_rates(x, y, y_pred, labelname,filename='default'):
    """This function plots both pressure and actual flow rates"""
    plt.figure()
    if labelname=='Training Data':
        color='red'
    else:
        color='green'

    plt.subplot(411)
    plot_pressure(x['TIME'], y, y_pred, labelname, color)

    plt.subplot(412)
    plot_rates(x['TIME'],x['WWPR:P1'],wellname='A',color='blue')
    plot_rates(x['TIME'],x['WOPR:P1'],wellname='A',color='green')

    plt.subplot(413)
    plot_rates(x['TIME'],x['WWPR:P2'],wellname='B',color='blue')
    plot_rates(x['TIME'],x['WOPR:P2'],wellname='B',color='green')


    plt.subplot(414)
    plot_rates(x['TIME'],x['WWPR:P3'],wellname='C',color='blue')
    plot_rates(x['TIME'],x['WOPR:P3'],wellname='C',color='green')
    plt.subplots_adjust(top=1.5,bottom=0.2)


def main():
    # Load Training and Test Set
    X_train,Y_train=load_data('https://raw.githubusercontent.com/titaristanto/data-driven-production-problem/master/far_lowWC_training.csv')
    X_test,y_test=load_data('https://raw.githubusercontent.com/titaristanto/data-driven-production-problem/master/close_highWC_test.csv')

    # Build and select features
    x_train_temp=structure_features(X_train)
    x_test=structure_features(X_test)

    # Split train and dev set
    x_train,x_dev,y_train,y_dev=train_test_split(x_train_temp,Y_train,test_size=0.25,shuffle=False)

    # Train the data
    start_time = time.time()
    clf = regression(x_train, y_train)

    # Predict features
    y_pred_train = clf.predict(x_train)
    y_pred_dev = clf.predict(x_dev)
    y_pred_test = clf.predict(x_test)

    # Measure running time
    print("Completed in %s seconds" % (time.time() - start_time))

    ## Error Calculation
    # Training Data Error Calculation
    print('Training Set Score: %1.4f' % (clf.score(x_train,y_train)))

    # Cross-val / Dev Set Error Calculation
    scores = cross_val_score(clf, x_train, y_train, cv=4)
    print("Dev Set Score (Cval): {:.3f} (std: {:.3f})".format(scores.mean(),scores.std()),end="\n\n" )
    # [float(score) for score in scores] # printing each fold score
    print('Dev Set Score: %1.4f' % (clf.score(x_dev,y_dev)))

    # Test Data Error Calculation
    print('Test Set Score: %1.4f' % (clf.score(x_test,y_test)))

    # Plot Training Data (P and Q)
    plot_pressure_rates(X_train,
                        pd.concat([y_train,y_dev],axis=0,join='inner'),
                        np.concatenate((y_pred_train,y_pred_dev),axis=0),
                        labelname='Training Data')

    # Plot Test Data (P and Q)
    plot_pressure_rates(X_test,y_test,y_pred_test,labelname='Test Data')

    # Pipeline Analysis
    alg_sum=build_pipe(X_train,Y_train,X_test,y_test)
    print(alg_sum)

    # Plot Derivatives
    #derivatives(clf)

    # Load Classifier from file
    # clf = joblib.load('filename.pkl')

if __name__ == '__main__':
    main()
