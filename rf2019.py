# -*- coding: utf-8 -*-
"""
Implementation of RF algorithm using CICDos2019 data set
Created on October 7 2020
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
from sklearn.metrics import accuracy_score, recall_score,f1_score,precision_score,roc_auc_score
from joblib import dump, load

from google.colab import drive 
drive.mount('/content/gdrive')

datasetTrain = pd.read_csv('gdrive/My Drive/Datasets/2019/dataTrain_NormalTraffic.csv')
datasetTest = pd.read_csv('gdrive/My Drive/Datasets/2019/dataTest_NormalTraffic.csv')
datasetTrain = datasetTrain.dropna()
datasetTest = datasetTest.dropna()

datasetTrain.drop(columns=['Unnamed: 0'])

datasetTest.drop(columns=['Unnamed: 0'])

# Separate data> Labels
Y_train = datasetTrain.loc[:,['Label']].values
Y_test = datasetTest.loc[:,['Label']].values

# Separate data> values
X_train = datasetTrain.loc[:, [ 'Flow.Duration', 'Tot.Fwd.Pkts',
       'TotLen.Fwd.Pkts', 'TotLen.Bwd.Pkts', 'Fwd.Pkt.Len.Max',
       'Fwd.Pkt.Len.Min', 'Fwd.Pkt.Len.Std', 'Bwd.Pkt.Len.Max',
       'Bwd.Pkt.Len.Min', 'Bwd.Pkt.Len.Mean', 'Bwd.Pkt.Len.Std', 'Flow.Byts.s',
       'Flow.IAT.Std', 'Flow.IAT.Max', 'Flow.IAT.Min', 'Fwd.IAT.Tot',
       'Fwd.IAT.Mean', 'Fwd.IAT.Max', 'Fwd.IAT.Min', 'Bwd.IAT.Tot',
       'Bwd.IAT.Mean', 'Bwd.IAT.Std', 'Bwd.IAT.Max', 'Bwd.IAT.Min',
       'Fwd.PSH.Flags', 'Fwd.Header.Len', 'Bwd.Header.Len', 'Fwd.Pkts.s',
       'Bwd.Pkts.s', 'Pkt.Len.Max', 'Pkt.Len.Std', 'Pkt.Len.Var',
       'FIN.Flag.Cnt', 'SYN.Flag.Cnt', 'RST.Flag.Cnt', 'PSH.Flag.Cnt',
       'ACK.Flag.Cnt', 'CWE.Flag.Count', 'ECE.Flag.Cnt', 'Down.Up.Ratio',
       'Pkt.Size.Avg', 'Bwd.Seg.Size.Avg', 'Bwd.Pkts.b.Avg',
       'Bwd.Blk.Rate.Avg', 'Subflow.Fwd.Pkts', 'Subflow.Fwd.Byts',
       'Subflow.Bwd.Byts', 'Init.Fwd.Win.Byts', 'Init.Bwd.Win.Byts',
       'Fwd.Act.Data.Pkts', 'Fwd.Seg.Size.Min', 'Idle.Mean', 'Idle.Std']].values
                            
X_test = datasetTest.loc[:, ['Flow.Duration', 'Tot.Fwd.Pkts',
       'TotLen.Fwd.Pkts', 'TotLen.Bwd.Pkts', 'Fwd.Pkt.Len.Max',
       'Fwd.Pkt.Len.Min', 'Fwd.Pkt.Len.Std', 'Bwd.Pkt.Len.Max',
       'Bwd.Pkt.Len.Min', 'Bwd.Pkt.Len.Mean', 'Bwd.Pkt.Len.Std', 'Flow.Byts.s',
       'Flow.IAT.Std', 'Flow.IAT.Max', 'Flow.IAT.Min', 'Fwd.IAT.Tot',
       'Fwd.IAT.Mean', 'Fwd.IAT.Max', 'Fwd.IAT.Min', 'Bwd.IAT.Tot',
       'Bwd.IAT.Mean', 'Bwd.IAT.Std', 'Bwd.IAT.Max', 'Bwd.IAT.Min',
       'Fwd.PSH.Flags', 'Fwd.Header.Len', 'Bwd.Header.Len', 'Fwd.Pkts.s',
       'Bwd.Pkts.s', 'Pkt.Len.Max', 'Pkt.Len.Std', 'Pkt.Len.Var',
       'FIN.Flag.Cnt', 'SYN.Flag.Cnt', 'RST.Flag.Cnt', 'PSH.Flag.Cnt',
       'ACK.Flag.Cnt', 'CWE.Flag.Count', 'ECE.Flag.Cnt', 'Down.Up.Ratio',
       'Pkt.Size.Avg', 'Bwd.Seg.Size.Avg', 'Bwd.Pkts.b.Avg',
       'Bwd.Blk.Rate.Avg', 'Subflow.Fwd.Pkts', 'Subflow.Fwd.Byts',
       'Subflow.Bwd.Byts', 'Init.Fwd.Win.Byts', 'Init.Bwd.Win.Byts',
       'Fwd.Act.Data.Pkts', 'Fwd.Seg.Size.Min', 'Idle.Mean', 'Idle.Std']].values

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
regressor = RandomForestClassifier(n_estimators=350, random_state=0, n_jobs=-1, min_samples_split=4, max_leaf_nodes=5)
# Train model
regressor.fit(X_train, Y_train)
# Apply trained model to test
y_pred = regressor.predict(X_test)

# demonstrate verify fitting on training
plt.plot(y_pred)
plt.plot(Y_train)

# demonstrate verify fitting on testing
plt.plot(y_pred)
plt.plot(Y_test)

dump(regressor, 'gdrive/My Drive/Datasets/2019/RF2019/RF2019.joblib')

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