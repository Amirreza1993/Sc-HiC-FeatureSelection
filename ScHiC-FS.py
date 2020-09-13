
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


#CELL LINE Dataset 
#DataMain=np.load("/home/nanni/Projects/SingleCellHiC/DI_indices/human_DIs.npy")
DataMain=np.load("/home/amirreza/DI_indices22/human_DIs-cL_FULL.npy")
data=DataMain
cell_line={'HAP1':1, 'HeLa':2,'GM12878':3, 'K562':4}
human_meta = pd.read_csv(os.path.join("./DI_indices", '/home/nanni/Projects/SingleCellHiC/DI_indices/human_meta.tsv'), sep="\t")
human_meta.cell_line=[cell_line[item] for item in human_meta.cell_line]
label=human_meta['cell_line']
label=np.array(label)

##CELL CYCLE Dataset
#DataMain=np.load("/home/amirreza/Dataset/MAIN/Dataset/human_DIs-cc.npy")
#data=DataMain
#cell_cycle={'G1':1, 'ES':2,'MS':3, 'G2':4}
#human_meta = pd.read_csv( '/home/amirreza/Dataset/MAIN/Dataset/human_meta_cellcycle.tsv', sep="\t")
#human_meta.cell_cycle=[cell_cycle[item] for item in human_meta.cell_cycle]
#label=human_meta['cell_cycle']
#label=np.array(label)


#X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=40)
#y_train=y_train.astype(int)
#y_test=y_test.astype(int)

i=0
acc=np.zeros(25)
ARI=np.zeros(25)


kf = RepeatedKFold(n_splits=5, n_repeats=5) 
X=data
y=label
for train_index, test_index in kf.split(X):
    
    X_train=X[train_index]
    X_test=X[test_index]
    y_train=y[train_index]
    y_test=y[test_index]
    imputer=SimpleImputer(strategy="median")
    imputer.fit(X_train)
    X_train=imputer.transform(X_train)
    imputer.fit(X_test)
    X_test=imputer.transform(X_test)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    X_test = sc.transform(X_test)

    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    num_pip=Pipeline([
        ('imputer',SimpleImputer(strategy="median")),
        ('std_scaler',StandardScaler()),
    ])

    X_train=num_pip.fit_transform(X_train)
    X_test = num_pip.transform(X_test)
    print('fs')
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
    #fscore
    from skfeature.function.statistical_based import f_score
    score = f_score.f_score(X_train, y_train)
    idx_fscore = f_score.feature_ranking(score)
    print(idx_fish[0:20])
    #Integration
    print('ens')
     x=1150 #Threshold for each FS method
    idxM=idx_fish[:x]
    idxN=idx_fish[:x]
    idxO=idx_fscore[:x]
    idxP=idx_lap[:x]
 #AND
    idx = set(idxO) - (set(idxO) - set(idxM)-set(idxN))
    idx_and=reduce(np.intersect1d, (idxO, idxM, idxN))
    print(idx_and.shape)
    #OR
    idx=np.concatenate((idxM,idxN))
    idx_or=np.unique(idx)
    print(idx_or.shape)
#MV
    idx=np.concatenate((idxM,idxN,idxO))
    c = Counter(idx)
    a=c.most_common()
    b=[i[0] for i in a if i[1]>1 ]
    idx_mv=np.array(b)
    print(idx_mv.shape)
##select fs
    #Random selection
    idx_rand=random.sample(range(0, np.shape(X_train)[1]), num_fea)


    idx=idx_rand #Integration method
    num_fea = 250
    selected_features_train=X_train[:,idx[:num_fea]]
    selected_features_test=X_test[:,idx[:num_fea]]
    print('cl')
#MLPC
    architecture = (100,50,50) ###(A num_layers sized tuple with number of hidden neurons as each element)
    activationf = 'relu'
    learning_rate=0.01
    #batch_size=200
    solver='adam'
    mlp = MLPClassifier(hidden_layer_sizes=architecture,activation=activationf,learning_rate_init=learning_rate,solver=solver, max_iter=400)
    mlp.fit(selected_features_train,y_train) #training classifier
    y_predict = mlp.predict(selected_features_test)
    acc[i] = accuracy_score(y_test, y_predict)
    y_score=mlp.predict_proba(selected_features_test)
    ARI[i]=adjusted_rand_score(y_test,y_predict)
    i=i+1
  
