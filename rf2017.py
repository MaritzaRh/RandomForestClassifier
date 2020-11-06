# -*- coding: utf-8 -*-
"""
Implementation of RF algorithm using CIC2017 data set
Created on October 16 2020
By Maritza Rosales H.
All rights reserved
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve, precision_recall_fscore_support
from matplotlib import pyplot as plt
import scikitplot as skplt
from joblib import dump, load

from google.colab import drive 
drive.mount('/content/gdrive')

datasetTrain = pd.read_csv('gdrive/My Drive/Datasets/2017/dataTrain.csv')
datasetTest = pd.read_csv('gdrive/My Drive/Datasets/2017/dataTest.csv')

datasetTrain.drop(columns=['Unnamed: 0'])

datasetTest.drop(columns=['Unnamed: 0'])

# Separate data> Labels
Y_train = datasetTrain.loc[:,['Label']].values
Y_test = datasetTest.loc[:,['Label']].values

# Separate data> values
X_train = datasetTrain.loc[:, ['Flow.Duration',	'Tot.Fwd.Pkts',	'Tot.Bwd.Pkts',  
                         'TotLen.Fwd.Pkts',	'TotLen.Bwd.Pkts', 'Fwd.Pkt.Len.Max',	
                         'Fwd.Pkt.Len.Min',	'Fwd.Pkt.Len.Std', 'Bwd.Pkt.Len.Max',
                         'Bwd.Pkt.Len.Min',	'Bwd.Pkt.Len.Std', 'Flow.Byts.s',	
                         'Flow.Pkts.s',	'Flow.IAT.Mean',	'Flow.IAT.Std',	
                         'Flow.IAT.Max',	'Flow.IAT.Min',	'Fwd.IAT.Mean',	
                         'Fwd.IAT.Std',	'Fwd.IAT.Min',	'Bwd.IAT.Tot',	
                         'Bwd.IAT.Mean',	'Bwd.IAT.Std',	'Bwd.IAT.Max',	
                         'Bwd.IAT.Min',	'Fwd.PSH.Flags',	'Fwd.Pkts.s',	
                         'Bwd.Pkts.s',	'Pkt.Len.Min',	'Pkt.Len.Max',
                         'Pkt.Len.Mean',	'Pkt.Len.Std',	'Pkt.Len.Var',
                         'FIN.Flag.Cnt',	'SYN.Flag.Cnt',	'RST.Flag.Cnt',
                         'PSH.Flag.Cnt',	'Down.Up.Ratio', 'Bwd.Pkts.b.Avg',	
                         'Bwd.Blk.Rate.Avg',	'Subflow.Fwd.Pkts',	'Subflow.Fwd.Byts',
                         'Subflow.Bwd.Byts',	'Init.Fwd.Win.Byts',	'Init.Bwd.Win.Byts',
                         'Fwd.Seg.Size.Min', 'Idle.Mean', 'Idle.Std',	'Idle.Min']].values
#'Src.IP',	'Src.Port',	'Dst.IP',	'Dst.Port', 'Protocol',	                         
X_test = datasetTest.loc[:, ['Flow.Duration',	'Tot.Fwd.Pkts',	'Tot.Bwd.Pkts',
                         'TotLen.Fwd.Pkts',	'TotLen.Bwd.Pkts', 'Fwd.Pkt.Len.Max',	
                         'Fwd.Pkt.Len.Min',	'Fwd.Pkt.Len.Std', 'Bwd.Pkt.Len.Max',
                         'Bwd.Pkt.Len.Min',	'Bwd.Pkt.Len.Std', 'Flow.Byts.s',	
                         'Flow.Pkts.s',	'Flow.IAT.Mean',	'Flow.IAT.Std',	
                         'Flow.IAT.Max',	'Flow.IAT.Min',	'Fwd.IAT.Mean',	
                         'Fwd.IAT.Std',	'Fwd.IAT.Min',	'Bwd.IAT.Tot',	
                         'Bwd.IAT.Mean',	'Bwd.IAT.Std',	'Bwd.IAT.Max',	
                         'Bwd.IAT.Min',	'Fwd.PSH.Flags',	'Fwd.Pkts.s',	
                         'Bwd.Pkts.s',	'Pkt.Len.Min',	'Pkt.Len.Max',
                         'Pkt.Len.Mean',	'Pkt.Len.Std',	'Pkt.Len.Var',
                         'FIN.Flag.Cnt',	'SYN.Flag.Cnt',	'RST.Flag.Cnt',
                         'PSH.Flag.Cnt',	'Down.Up.Ratio', 'Bwd.Pkts.b.Avg',	
                         'Bwd.Blk.Rate.Avg',	'Subflow.Fwd.Pkts',	'Subflow.Fwd.Byts',
                         'Subflow.Bwd.Byts',	'Init.Fwd.Win.Byts',	'Init.Bwd.Win.Byts',
                         'Fwd.Seg.Size.Min', 'Idle.Mean', 'Idle.Std',	'Idle.Min']].values

# Feature scaling> standarize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#PCA for 85% 
pca = PCA(0.85)
pca.fit(X_train)
pca.n_components_

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
Y_train = np.ravel(Y_train,order='C')
Y_test = np.ravel(Y_test,order='C')

# Create model
regressor = RandomForestClassifier(n_estimators=750, random_state=10, max_depth=25, n_jobs=-1,criterion='entropy', max_features=15)
# Train model
regressor.fit(X_train, Y_train)
# Apply trained model to test
y_pred = regressor.predict(X_test)

dump(regressor, 'gdrive/My Drive/Datasets/2017/RF2017/RF2017.joblib')

# demonstrate verify fitting on training
plt.plot(y_pred)
plt.plot(Y_train)

# demonstrate verify fitting on testing
plt.plot(y_pred)
plt.plot(Y_test)

#TWO CLASSES 
TP=TN=FP=FN=0
for k in range (0,len(Y_test)):
    ylabel     = Y_test[k];
    ypredicted = y_pred[k];
    if ypredicted == 0 and ylabel == 0:
        TN = TN + 1
    elif ypredicted > 0 and ylabel > 0:
        TP = TP + 1
    elif ypredicted > 0 and ylabel == 0:
        FP = FP + 1
    elif ypredicted == 0 and ylabel > 0:
        FN = FN + 1
    else:
        print('any')
print("ACCURACY", (TP+TN)/(TP+TN+FP+FN))
print("F1-SCORE", (2*TP)/(2*TP+FP+FN))
print("FALSE POS. RATE (FPR)", (FP)/(FP+TN))
print("RECALL (TPR)", (TP)/(TP+FN))
print("Precision ", (TP)/(TP+FP))

#MULTIPLE CLASSES
print("Accuracy", accuracy_score(Y_test, y_pred))
print("Precision", precision_score(Y_test, y_pred, average='weighted'))
print("F1-SCORE", f1_score(Y_test, y_pred, average='weighted'))
print("RECALL (TPR)", recall_score(Y_test, y_pred, average='weighted'))

#print(confusion_matrix(Y_test,y_pred, normalize='true'))
skplt.metrics.plot_confusion_matrix(Y_test,y_pred,normalize="True")