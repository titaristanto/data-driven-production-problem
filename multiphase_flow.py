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
from sklearn.ensemble import RandomForestRegressor
from scipy import special
import scipy
import matplotlib.pyplot as plt
import time, os, math, warnings, xlsxwriter

def load_data(filename):
    os.chdir('C:\\Users\\E460\\PycharmProjects\\untitled3\\Research\\csv files')
    df = pd.read_csv(filename)

    t=df.loc[:,['TIME']] # Time in simulation: DAY
    t*=24 # Converting time from DAY to HOUR
    qo=df.loc[:,['WOPR:P1', 'WOPR:P2', 'WOPR:P3']]
    qw=df.loc[:,['WWPR:P1', 'WWPR:P2', 'WWPR:P3']]
    p=df.loc[:,['WBHP:P1']]
    wc=df.loc[:,['WWCT:P1', 'WWCT:P2', 'WWCT:P3']]
    x=pd.concat([t,qo,qw,wc],axis=1,join='inner')
    return x,p

def buildfeature(tset,qset):
    warnings.filterwarnings("ignore")

    k=50            # Perm in mD
    por=0.2         # Porosity in fraction
    m=2             # Viscosity in cP
    re=1000      # Re in ft
    c=10**-5       # Total compressibility in 1/psi

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
    for i in range(1,numdata):
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
    f2tot=np.sum(f2,axis=1)
    f3tot=np.sum(f3,axis=1)
    f4tot=np.sum(f4,axis=1)
    f5tot=np.sum(f5,axis=1)
    f6tot=np.sum(f6,axis=1)
    f7tot=np.sum(f7,axis=1)
    f8tot=np.sum(f8,axis=1)
    f9tot=np.sum(f9,axis=1)
    f10tot=np.sum(f10,axis=1)

    features=[0,q,f2tot, f3tot, f4tot, f5tot, f6tot, f7tot, f8tot, f9tot, f10tot]

    return features

def regression(x_train, y_train):
    ### import the sklearn regression module, create, and train the linear regression func
    from sklearn.linear_model import Ridge

    #clf=linear_model.LinearRegression()
    #clf=Ridge(alpha=1)
    clf=GridSearchCV(Ridge(),[{'alpha':[0.01,0.1,1,10,100,1000]}],cv=3)

    clf.fit(x_train, y_train)
    #print(clf.coef_)
    return clf

def build_pipe(x_train,y_train,x_test,y_test):
    cval_list=[]
    clf_list=[]
    classifiers=[Ridge(),
                 KernelRidge(),
                 SGDRegressor(),
                 SVR(),
                 DecisionTreeRegressor(),
                 RandomForestRegressor()]
    param_grid=[{'clf__alpha':[0.1,1,2,5,10]},
                [{'clf__alpha':[1,10,100,1000],'clf__kernel':['linear']},{'clf__alpha':[1,10,100,1000],'clf__kernel':['polynomial'],'clf__degree':[2,3,5]}],
                {'clf__alpha':[0.001,0.01,0.1,1]},
                [{'clf__C':[0.01,0.1,1,10,100],'clf__kernel':['linear','poly','rbf']}],
                {'clf__max_depth': [None, 5, 10, 30, 50]},
                [{'clf__n_estimators':[2, 3, 5, 10, 20],'clf__max_depth': [None, 5, 10, 20, 30, 50]}]]
    for i in range(len(classifiers)):
        pipe=Pipeline([('scl', StandardScaler()),
                       ('clf',classifiers[i])])
        clf=GridSearchCV(pipe,cv=3,param_grid=param_grid[i])
        clf.fit(x_train, y_train)
        clf_list.append(clf.best_estimator_)
        cval_list.append(clf.best_score_)

    # Compare accuracies
    pipe_dict = {0: 'Ridge Regression',
                 1: 'Kernel Ridge Regression',
                 2: 'SGD Regression',
                 3: 'Support Vector Regression',
                 4: 'Decision Tree',
                 5: 'Random Forest'}
    train_score=[]
    test_score=[]
    for idx, val in enumerate(clf_list):
        train_score.append(val.score(x_train, y_train))
        test_score.append(val.score(x_test, y_test))
        print('%s training set accuracy: %.3f' % (pipe_dict[idx], train_score[idx]))
        print('%s dev set accuracy: %.3f' % (pipe_dict[idx], cval_list[idx]))
        print('%s test set accuracy: %.3f \n' % (pipe_dict[idx], test_score[idx]))

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

    # Constructing DataFrame output
    dict_sum=OrderedDict([('Algorithms',[i for i in pipe_dict.values()]),
                          ('Training Set Accuracy',train_score),
                          ('Dev Set Accuracy',cval_list),
                          ('Test Set Accuracy',test_score)])
    df=pd.DataFrame.from_dict(dict_sum)

    writer = pd.ExcelWriter('C:\\Users\\E460\\PycharmProjects\\untitled3\\Research\\results\\summary_report_temp.xlsx', engine='xlsxwriter')
    df.to_excel(writer, index=False)
    writer.save()
    return df

def derivatives(t,p_actu,p_pred):
    # Delta Pressure
    p_act=p_actu.as_matrix()
    dp_act=abs(p_act[0]-p_act)
    dp_pred=abs(p_pred[0]-p_pred)

    # Derivatives
    p_der_act=np.zeros(len(p_act)-2)
    p_der_pred=np.zeros(len(p_pred)-2)
    for i in range(1,len(p_act)-1):
        p_der_act[i-1]=t[i]/(t[i+1]-t[i-1])*(dp_act[i+1]-dp_act[i-1])
        p_der_pred[i-1]=t[i]/(t[i+1]-t[i-1])*(dp_pred[i+1]-dp_pred[i-1])
    plot_derivatives(t,dp_act,dp_pred,p_der_act,p_der_pred)

def plot_derivatives(t,dp_act,dp_pred,p_der_act,p_der_pred):
    plt.figure()
    plt.loglog(t, dp_act, 'ko',linewidth=3,label='Actual dP')
    plt.loglog(t, dp_pred, 'mx',label='Predicted dP')
    plt.loglog(t[1:-1], p_der_act, 'ko',linewidth=3,label='Actual Derivative')
    plt.loglog(t[1:-1], p_der_pred, 'mx',label='Predicted Derivative')

def plot_pressure(t,p_actual,p_pred,title,color):
    # Plotting pwf v time
    plt.plot(t, p_actual, 'k-',linewidth=3,label='Actual Pwf')

    if title=='Training Data':
        plt.plot(t[0:int(0.7*p_pred.shape[0])], p_pred[0:int(0.7*p_pred.shape[0])], 'rx',markeredgecolor=color,label=title)
        plt.plot(t[int(0.7*p_pred.shape[0]):], p_pred[int(0.7*p_pred.shape[0]):], 'yx',markeredgecolor='orange',label='Dev Set')
    else:
        plt.plot(t, p_pred, 'rx',markeredgecolor=color,label=title)
    plt.xlabel("Time (hours)")
    plt.ylabel("BH Pressure (psi)", fontsize=12)
    plt.title("BH Pressure Well A",y=0.9, fontsize=15)
    plt.legend(loc="best", prop=dict(size=12))
    plt.ylim(0, max(max(p_actual.values),max(p_pred)))
    plt.grid(True)

def plot_rates(t,q,wellname,color):
    # Plotting Flow Rate v time
    plt.plot(t, q,color, linewidth=3)
    plt.xlabel("Time (hours)")
    plt.ylabel("Flow Rate (STB/D)", fontsize=12)
    plt.title('Flow Rate Well '+wellname,y=0.86, fontsize=15)
    plt.xlim(0,max(t))
    plt.ylim(0,max(q))
    plt.grid(True)

def plot_pressure_rates(x, y, y_pred, labelname):
    plt.figure()
    plt.subplot(411)
    plot_pressure(x['TIME'], y, y_pred, labelname, 'red')

    plt.subplot(412)
    plot_rates(x['TIME'],x['WOPR:P1'],wellname='A',color='green')
    plot_rates(x['TIME'],x['WWPR:P1'],wellname='A',color='blue')

    plt.subplot(413)
    plot_rates(x['TIME'],x['WOPR:P2'],wellname='B',color='green')
    plot_rates(x['TIME'],x['WWPR:P2'],wellname='B',color='blue')

    plt.subplot(414)
    plot_rates(x['TIME'],x['WOPR:P3'],wellname='C',color='green')
    plot_rates(x['TIME'],x['WWPR:P3'],wellname='C',color='blue')

def main():
    # Loading Training and Test Set
    X_train,Y_train=load_data('far_lowWC_training.csv')
    X_test,y_test=load_data('far_lowWC_test.csv')

    t_train=X_train['TIME']
    t_test=X_test['TIME']

    # Building features - Training Set
    fa_tr=buildfeature(t_train,X_train['WOPR:P1'])
    fb_tr=buildfeature(t_train,X_train['WOPR:P2'])
    fc_tr=buildfeature(t_train,X_train['WOPR:P3'])
    fa_wr_tr=buildfeature(t_train,X_train['WWPR:P1'])
    fb_wr_tr=buildfeature(t_train,X_train['WWPR:P2'])
    fc_wr_tr=buildfeature(t_train,X_train['WWPR:P3'])

    # Building features - Test Set
    fa_te=buildfeature(t_test,X_test['WOPR:P1'])
    fb_te=buildfeature(t_test,X_test['WOPR:P2'])
    fc_te=buildfeature(t_test,X_test['WOPR:P3'])
    fa_wr_te=buildfeature(t_test,X_test['WWPR:P1'])
    fb_wr_te=buildfeature(t_test,X_test['WWPR:P2'])
    fc_wr_te=buildfeature(t_test,X_test['WWPR:P3'])

    # Selecting features
    #fA=pd.DataFrame({'f1':fa1, 'f2':fa2, 'f3':fa3, 'f4':fa4, 'fb1':fb1, 'fb2':fb2, 'fb3':fb3, 'fb4':fb4,'fb9':fb9,'fc1':fc1, 'fc2':fc2, 'fc3':fc3, 'fc4':fc4,'fc9':fc9})
    #fA=pd.DataFrame({'f1':fa1, 'f2':fa2, 'f3':fa3, 'f4':fa4, 'fb1':fb1, 'fb2':fb2, 'fb3':fb3, 'fb4':fb4,'fb6':fb6,'fb7':fb7, 'fc1':fc1, 'fc2':fc2, 'fc3':fc3, 'fc4':fc4,'fc6':fc6,'fc7':fc7})
    x_train_temp=pd.DataFrame({'fa1':fa_tr[1], 'fa2':fa_tr[2], 'fa3':fa_tr[3], 'fa4':fa_tr[4],'fa10':fa_tr[10],
                               'fb1':fb_tr[1], 'fb2':fb_tr[2], 'fb3':fb_tr[3], 'fb4':fb_tr[4],'fb10':fb_tr[10],
                               'fc1':fc_tr[1], 'fc2':fc_tr[2], 'fc3':fc_tr[3], 'fc4':fc_tr[4],'fc10':fc_tr[10],
                               'faw1':fa_wr_tr[1], 'faw2':fa_wr_tr[2], 'faw3':fa_wr_tr[3], 'faw4':fa_wr_tr[4],'faw10':fa_wr_tr[10],
                               'fbw1':fb_wr_tr[1], 'fbw2':fb_wr_tr[2], 'fbw3':fb_wr_tr[3], 'fbw4':fb_wr_tr[4],'fbw10':fb_wr_tr[10],
                               'fcw1':fc_wr_tr[1], 'fcw2':fc_wr_tr[2], 'fcw3':fc_wr_tr[3], 'fcw4':fc_wr_tr[4],'fcw10':fc_wr_tr[10]})
    x_test=pd.DataFrame({'fa1':fa_te[1], 'fa2':fa_te[2], 'fa3':fa_te[3], 'fa4':fa_te[4],'fa10':fa_te[10],
                         'fb1':fb_te[1], 'fb2':fb_te[2], 'fb3':fb_te[3], 'fb4':fb_te[4],'fb10':fb_te[10],
                         'fc1':fc_te[1], 'fc2':fc_te[2], 'fc3':fc_te[3], 'fc4':fc_te[4],'fc10':fc_te[10],
                         'faw1':fa_wr_te[1], 'faw2':fa_wr_te[2], 'faw3':fa_wr_te[3], 'faw4':fa_wr_te[4],'faw10':fa_wr_te[10],
                         'fbw1':fb_wr_te[1], 'fbw2':fb_wr_te[2], 'fbw3':fb_wr_te[3], 'fbw4':fb_wr_te[4],'fbw10':fb_wr_te[10],
                         'fcw1':fc_wr_te[1], 'fcw2':fc_wr_te[2], 'fcw3':fc_wr_te[3], 'fcw4':fc_wr_te[4],'fcw10':fc_wr_te[10]})

    x_train,x_dev,y_train,y_dev=train_test_split(x_train_temp,Y_train,test_size=0.3,shuffle=False)

    # Train the data
    start_time = time.time()
    clf = regression(x_train, y_train)

    # Prediction Result
    y_pred_train=clf.predict(x_train)
    y_pred_dev=clf.predict(x_dev)
    y_pred_test=clf.predict(x_test)

    # Measuring running time
    print("Completed in %s seconds" % (time.time() - start_time))

    ## Error Calculation
    # Training Data Error Calculation
    print('Training Set Score: %1.4f' % (clf.score(x_train,y_train)))

    # Cross-val / Dev Set Error Calculation
    scores=cross_val_score(clf, x_train, y_train, cv=3)
    print("Dev Set Score: {:.3f} (std: {:.3f})".format(scores.mean(),scores.std()),end="\n\n" )
    # [float(score) for score in scores] # printing each fold score

    # Test Data Error Calculation
    print('Test Set Score: %1.4f' % (clf.score(x_test,y_test)))

    # Plotting Training Data (P and Q)
    plot_pressure_rates(X_train,pd.concat([y_train,y_dev],axis=0,join='inner'),np.concatenate((y_pred_train,y_pred_dev),axis=0),labelname='Training Data')

    # Plotting Test Data (P and Q)
    plot_pressure_rates(X_test,y_test,y_pred_test,labelname='Test Data')

    # Pipeline Analysis
    alg_sum=build_pipe(x_train_temp,Y_train,x_test,y_test)
    print(alg_sum)

    # Plot Derivatives
    derivatives(t_train[0:len(x_train)],y_train,y_pred_train)


if __name__ == '__main__':
    main()
