import random
from skfeature.function.similarity_based import reliefF
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from skfeature.utility import unsupervised_evaluation
from skfeature.function.similarity_based import fisher_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold , RepeatedKFold
from sklearn.preprocessing import StandardScaler
from functools import reduce
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import os
import pandas as pd 
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
####################################IMPORT DATASET #######################3
#CELL LINE Dataset 
data=np.load("") # location of .npy sample files
cell_line={'HAP1':1, 'HeLa':2,'GM12878':3, 'K562':4}
matrices_meta = pd.read_csv(os.path.join('matrices_meta/human_meta_cell-line.tsv'), sep="\t")
matrices_meta.cell_line=[cell_line[item] for item in matrices_meta.cell_line]
label=matrices_meta['cell_line']
label=np.array(label)

#CELL CYCLE Dataset
DataMain=np.load("")# location of .npy sample files
cell_cycle={'G1':1, 'ES':2,'MS':3, 'G2':4}
matrices_meta = pd.read_csv( 'matrices_meta/human_meta_cellcycle.tsv', sep="\t")
matrices_meta.cell_cycle=[cell_cycle[item] for item in matrices_meta.cell_cycle]
label=matrices_meta['cell_cycle']
label=np.array(label)

combination_method=1 #For AND operation :1 ,OR Operation :2, Majority Voting Operation :3 and Random selection 4
threshold=1150 #Threshold for each feature selection method


i=0
acc=np.zeros(25) #to store the results of Accuracy
ARI=np.zeros(25)#to store the results of ARI
YP=[]
YT=[]
YS=[]
# Use 5 fold cross validation
kf = RepeatedKFold(n_splits=5, n_repeats=5) 
X=data
y=label
for train_index, test_index in kf.split(X):
    
    X_train=X[train_index]
    X_test=X[test_index]
    y_train=y[train_index]
    y_test=y[test_index]
    
######################### Data Preprocessing 
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    num_pip=Pipeline([
        ('imputer',SimpleImputer(strategy="median")),
        ('std_scaler',StandardScaler()),
    ])

    X_train=num_pip.fit_transform(X_train)
    X_test = num_pip.transform(X_test)
    print('fs')
    
########################### Apply Feature Selection methods :ReliefF, Laplacian score & Fisher
    #ReliefF
    score_rel = reliefF.reliefF(X_train, y_train)
    idx_rel = reliefF.feature_ranking(score_rel)
    #Laplacian score
    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "k": 7, 't': 1,'reliefF':True}
    W = construct_W.construct_W(X_train, **kwargs_W)
    score_lap = lap_score.lap_score(X_train, W=W)
    idx_lap = lap_score.feature_ranking(score_lap)
    #Fisher
    score_fish = fisher_score.fisher_score(X_train, y_train)
    print(score_fish)
    idx_fish = fisher_score.feature_ranking(score_fish)
    ###################################### Feature Integration
    idxM=idx_rel[:threshold]
    idxN=idx_lap[:threshold]
    idxO=idx_fish[:threshold]

    if combination_method==1 :
    #AND
        idx_and=reduce(np.intersect1d, (idxO, idxM, idxN))
        idx=idx_and
        print("number of selectes features (bins) = ", idx.shape[0])

    if combination_method==2 :
    #OR
        idx=np.concatenate((idxM,idxN,idxO))
        idx=np.unique(idx)
        print("number of selectes features (bins) = ", idx.shape[0])


    if combination_method==3 :
    #MV
        idx=np.concatenate((idxM,idxN,idxO))
        c = Counter(idx)
        a=c.most_common()
        b=[i[0] for i in a if i[1]>1 ]
        idx=np.array(b)
        print("number of selectes features (bins) = ", idx.shape[0])

    if combination_method==4 :
    #Random selection
        idx=random.sample(range(0, np.shape(X_train)[1]), threshold)
        print("number of selectes features (bins) = ", idx.shape[0])

    selected_features_train=X_train[:,idx]
    selected_features_test=X_test[:,idx]
###################################### Implementing the MLP classifier
#MLPC
    architecture = (100,50,50) ###(A num_layers sized tuple with number of hidden neurons as each element)
    activationf = 'relu'
    learning_rate=0.01
    #batch_size=200
    solver='adam'
    mlp = MLPClassifier(hidden_layer_sizes=architecture,activation=activationf,learning_rate_init=learning_rate,solver=solver, max_iter=400)
    mlp.fit(selected_features_train,y_train) #training classifier
    YT.append( y_test)
    y_predict = mlp.predict(selected_features_test)
    YP.append( y_predict)
    acc[i] = accuracy_score(y_test, y_predict)
    y_score=mlp.predict_proba(selected_features_test)
    YS.append(y_score)
    ARI[i]=adjusted_rand_score(y_test,y_predict)
    i=i+1
