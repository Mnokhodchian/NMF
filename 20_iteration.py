from sklearn import metrics
import numpy as np
import scipy.sparse.csgraph  as csgraph
import sklearn.neighbors as KNN
import scipy.linalg as la
import scipy as sp
import sklearn.preprocessing
from pymanopt.manifolds import Grassmann , stiefel
import pymanopt
from pymanopt.optimizers import TrustRegions
from sklearn.preprocessing import normalize
import sklearn.cluster as skcl
from PIL import Image
import glob
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.datasets import fetch_openml

import warnings
from sklearn.preprocessing import MinMaxScaler
import time

warnings.filterwarnings("ignore")
import math

def stiefel_manifold(W, L, dim, Grass_max_iter , teta):
    # # ############################## normalize L=I-D(-1/2) W D(-1/2)
    # with sp.errstate(divide="ignore"):
    #     D_sqrt = 1.0 / np.sqrt(D)
    # D_sqrt[np.isinf(D_sqrt)] = 0
    # L_nrmzed = np.dot(D_sqrt, np.dot(W, D_sqrt))
    # L = np.identity(len(L_nrmzed)) - L_nrmzed
    ########################################################  find the first k eigen vector of L corresponding to the smallest k eigenvalues
    eigvals, eigvecs = la.eigh(W, eigvals=(0, dim - 1))  # 2 cluster --> 0 and 1
    U0 = eigvecs
    manifold = stiefel.Stiefel(len(eigvecs), len(eigvecs[0]))

    @pymanopt.function.autograd(manifold)
    def cost(U):
        return np.trace(U.T @ L @ U)  # + 0.001 * np.trace(U @ U.T )#,ord=1)

    problem = pymanopt.Problem(manifold, cost)
    optimizer = TrustRegions(max_iterations=Grass_max_iter)
    result = optimizer.run(problem, initial_point=U0)  # , Delta_bar=8 * np.sqrt(dim))
    return result.point

def init_of_WH(points , n_cluster):
    ww = np.random.random((len(points),n_cluster))
    hh = np.random.random((n_cluster,len(points[0]) ))
    max_row= np.max(hh,axis=0)
    for i in range (len(hh[0])):
        hh[:,i] = np.where( hh[:,i]< max_row[i], 0, max_row[i])
    hh = normalize(hh, axis=1, norm='l2')
    # print(hh @ hh.T)
    return (ww,hh)
def NMF_decompose_old(points, n_cluster ): #,alpha , beta , lapp , alphaw=0 ,alphah=0 , l1_ratio=0  ): #, beta ,weight , diag , landa ):
    from sklearn.decomposition import NMF as NMF_old
    k = points.min()
    if k < 0:
        new_points = points + abs(k)
    else:
        new_points = points
    model = NMF_old(n_components=n_cluster , solver='mu' , max_iter=50000)# ,init='custom' )
    # ww = np.random.random((len(points),n_cluster))
    # hh = np.random.random((n_cluster,len(points[0]) ))
    UF = model.fit_transform(X=new_points )#, W=ww , H=hh) #, beta=beta , weight=weight , diag=diag , landa=landa, W=ww )
   # print('reconstruction_err=', model.reconstruction_err_, ' iteration=', model.n_iter_, ' n_features_in_=',
   #       model.n_features_in_)
    UF = np.around(UF, 5)
    return UF
def NMF_decompose(points, n_cluster ,alpha , beta , landa , diag , weight  , alphaw=0 ,alphah=0 , l1_ratio=0  ): #, beta ,weight , diag , landa ):
    from sklearn.my_decomposition import NMF
    k = points.min()
    if k < 0:
        new_points = points + abs(k)
    else:
        new_points = points
    model = NMF(n_components=n_cluster , solver='mu' , max_iter=50000 ,init='custom')#, alpha_W=alphaw, alpha_H=alphah ,l1_ratio=l1_ratio )
    ww,hh =init_of_WH(points,n_cluster)
    UF = model.fit_transform(X=new_points ,alpha=alpha ,beta=beta , landa=landa , diag=diag , weight=weight  , W=ww , H=hh) #, beta=beta , weight=weight , diag=diag , landa=landa, W=ww ) # , W=ww, H=hh)
    #print('reconstruction_err=', model.reconstruction_err_, ' iteration=', model.n_iter_, ' n_features_in_=',
    #      model.n_features_in_)
    UF = np.around(UF, 5)
    return UF
def NMF_decompose_lnmfs(points, n_cluster ,alpha , beta , landa , diag , weight ): #,alpha , beta , lapp , alphaw=0 ,alphah=0 , l1_ratio=0  ): #, beta ,weight , diag , landa ):
    from sklearn.LNMFS_decomposition import NMF as  LNMFS
    k = points.min()
    if k < 0:
        new_points = points + abs(k)
    else:
        new_points = points
    model = LNMFS(n_components=n_cluster , solver='mu' , max_iter=50000)# ,init='custom' )
    ww, hh = init_of_WH(points, n_cluster)
    UF = model.fit_transform(X=new_points , alpha=alpha , beta=beta , landa=landa , diag=diag , weight=weight  , W=ww , H=hh) #, beta=beta , weight=weight , diag=diag , landa=landa, W=ww )
   # print('reconstruction_err=', model.reconstruction_err_, ' iteration=', model.n_iter_, ' n_features_in_=',
   #       model.n_features_in_)
    UF = np.around(UF, 5)
    return UF
def load_data(dataset):
    if dataset == 0:  # coil100 dataset 7200*1024 #Classes 100
        n_cluster =  100
        n_sample = 7200
        n_neighbors = 8  # round(math.sqrt(n_sample))
        arr = np.loadtxt("C:\mohsen\paper 2\codes\datasets\coil100.csv",delimiter=",", dtype=int)
        X1=arr[:,0:1024]
        true_class= arr[:, 1024]
        print("coil100 Dataset")
    elif dataset==1:   #breastw dataset - Samples * attrib. 683 * 9 # cluster =2 # 239:malignant  other :benign
        n_cluster =  2
        n_sample = 683
        n_neighbors = 18 #round(math.sqrt(n_sample))
        arr = np.loadtxt(".\datasets\\breastw.csv",delimiter=",", dtype=int)
        y=arr[:,0:9].max(axis=0)
        X1 = arr[:,0:9] / y
        true_class= arr[:, 9]
        print("breast Dataset")
    elif dataset==2 :  #  pendigit dataset -  477 * 16 # cluster = 3
        n_cluster =  10 #3
        n_sample = 10992 #477
        n_neighbors = 33 #13 #round(math.sqrt(n_sample))  # 23
        arr = np.loadtxt("C:\mohsen\paper 2\codes\datasets\pendigits.csv", delimiter=",", dtype=int)
        # arr = np.loadtxt(".\datasets\\pendigits_389(477_17).csv", delimiter=",", dtype=int)
        X1 = arr[:, 0:16]
        # X1=X1/100
        true_class = arr[:,16]  #np.floor(arr[:, 16] / 3) - 1
        print("pendigit Dataset")
    elif dataset == 3:  # yale dataset #Classes 15 ,  Samples * Dim       165 * 1024
        n_cluster =  15
        n_sample = 165
        n_neighbors = 3  # round(math.sqrt(n_sample))
        arr = np.loadtxt(".\datasets\yale.csv", delimiter=",", dtype=int)
        y = arr[:, 0:1024].max(axis=0)
        X1 = arr[:, 0:1024] / y
        true_class = arr[:, 1024]
        print("yale Dataset")
    elif dataset==4 :   # optdigits dataset  # 5620 * 64  # cluster = 10
        n_cluster =  10
        n_sample = 5620
        n_neighbors = 24 #round(math.sqrt(n_sample))    #75
        arr = np.loadtxt(".\datasets\optdigits.csv",delimiter=",", dtype=int)
        y = arr[:,0:64].max(axis=0)
        y[y[:] == 0] = 1
        X1 = arr[:,0:64] / y
        true_class= arr[:, 64]
        print("optidigits Dataset")
    elif dataset==5 :    # coil20 dataset #Classes 20 ,    1440 * 1024
        n_cluster = 20
        n_sample = 1440
        n_neighbors = 8 #round(math.sqrt(n_sample))
        X1 = np.loadtxt(".\datasets\coil20.csv",delimiter=",", dtype=float)
        true_class = [[i] * 72 for i in range(20)]
        true_class = np.asarray(true_class).reshape(1440 * 1)
        print("Coil20 Dataset")
    elif dataset == 6:  # ORL  dataset  : 400 * 10304  class 40
        n_cluster =  40
        n_sample = 400
        n_neighbors = 3 #round(math.sqrt(n_sample))    #10
        arr = np.loadtxt(".\datasets\orl.csv",delimiter=",", dtype=int)
        y = arr.max(axis=0)
        y[y[:] == 0] = 1
        X1 = arr / y
        true_class= [[i] * 10 for i in range(40)]
        true_class=np.array(true_class).reshape([400 ,])
        print("ORL Dataset")
    elif dataset ==7 :      #yeast 1298 * 8  , classes 4  gene  , n_sample : 243 ,429 , 463 ,163
        n_cluster =  4
        n_sample = 1298
        n_neighbors =  13  #18
        arr = np.loadtxt(".\datasets\yeast.csv",delimiter=",", dtype=float)
        y = arr[:,0:8].max(axis=0)
        y[y[:] == 0] = 1
        X1 = arr[:,0:8] / y
        true_class= arr[:, 8]
        print("yeast Dataset")
    elif dataset ==8 :      #ecoli  327 * 7  , classes 5  gene , n_sample :143 ,77 , 35 ,20 52
        n_cluster =  5
        n_sample = 327
        n_neighbors =  4
        arr = np.loadtxt(".\datasets\ecoli.csv",delimiter=",", dtype=float)
        y = arr[:,0:7].max(axis=0)
        y[y[:] == 0] = 1
        X1 = arr[:,0:7] / y
        true_class= arr[:, 7]
        print("Ecoli Dataset")
    elif dataset == 9:  # Isolet  6238 * ( 617 +1 )  , classes 26  speech  , n_sample/class :  240  -1<num <1
        n_cluster = 26
        n_sample = 6238
        n_neighbors = 15
        arr = np.loadtxt(".\datasets\isolet.csv", delimiter=",", dtype=float)
        X1 = arr[:, 0:617]
        X1=MinMaxScaler().fit_transform(X1)
        true_class = arr[:, 617]
        print("Isolet Dataset")
    elif dataset == 10:  # CNAE_9  1080*(1+857)  , classes 9     n_sample/class : 11
        X1, true_class = fetch_openml('cnae-9', as_frame=False, return_X_y=True)
        true_class = true_class.astype(int)
        X1 = MinMaxScaler().fit_transform(X1)
        X1 = np.column_stack((X1, true_class))    # make dataset file for matlab
        np.savetxt(".\\datasets\\cnae.csv", X1, delimiter=',')
        n_cluster = 9
        n_sample = 1080
        n_neighbors = 11
        print("CNAE_9  Dataset")
    elif dataset == 11:  # USPS  9298 * 256   , classes 10     n_sample/class :  different
        # X1, true_class = fetch_openml(data_id=41082, as_frame=False, return_X_y=True)
        # true_class = true_class.astype(int)
        # X1 = MinMaxScaler().fit_transform(X1)
        # X1 = np.column_stack((X1, true_class))    # make dataset file for matlab
        # np.savetxt(".\\datasets\\USPS.csv", X1, delimiter=',')
        n_cluster = 10
        n_sample = 9298  # 256 feature
        n_neighbors = 30
        arr = np.loadtxt("C:\mohsen\paper 2\codes\datasets\\USPS.csv",delimiter=",", dtype=float)
        X1 = arr[:, 0:256] #/ y
        true_class = arr[:, 256] -1

        print("USPS handwritting   Dataset")
    elif dataset == 12:  # HANDWRITTEN MNIST  70000*784   , classes 10     n_sample/class :
        X1, true_class = fetch_openml('mnist_784', as_frame=False, return_X_y=True)
        true_class = true_class.astype(int)
        X1 = MinMaxScaler().fit_transform(X1)
        n_cluster = 10
        n_sample = 70000
        n_neighbors = 84
        print("MNIST  Dataset")
    elif dataset ==13 :         #mpeg7
        n_cluster = 70
        n_sample = 1400
        n_neighbors = 4 #round(math.sqrt(n_sample))
        # n_feat_neighbors = 8
        X1 = np.loadtxt(".\datasets\mpeg7.csv",delimiter=",", dtype=int)
        true_class = [[i] * 20 for i in range(70)]
        true_class = np.asarray(true_class).reshape(1400 * 1)
        # datasetname="mpeg7"
        print("mpeg7 Dataset")

    return (X1 , true_class,n_cluster ,n_sample ,n_neighbors )
def old_nmf_on_org_data(org_data, n_cluster ,true_class):
    print(" old_nmf_on_org_data :")
    NMF_result = NMF_decompose_old(org_data, n_cluster)  # , beta=beta, weight=weight, diag=diag, landa=landa2)
    pred_class = NMF_result.argmax(axis=1)
    contingency_matrix = metrics.cluster.contingency_matrix(true_class, pred_class)
    purity_of_old_NMF = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    print("%f %f" % purity_of_old_NMF , metrics.rand_score(true_class, pred_class))

    kmean_result = skcl.k_means(X=NMF_result, n_clusters=n_cluster)
    kmean_TP_points = np.column_stack((true_class, kmean_result[1]))
    contingency_matrix = metrics.cluster.contingency_matrix(true_class, pred_class)
    purity_of_old_NMF_and_kmean = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    print("%f %f " %purity_of_old_NMF_and_kmean , metrics.rand_score(true_class, kmean_result[1]))
def MY_decomp(data ,n_cluster ,true_class, diag , weight ):
            global kk, result , row , alpha , beta , end_time
            NMF_result = NMF_decompose(data, n_cluster, alpha=alpha, beta=beta, landa=0, diag=diag , weight=weight)
            #   without kmean
            # pred_class = NMF_result.argmax(axis=1)
            # NMF_TP_points = np.column_stack((true_class, pred_class))
            # purity_of_GRass_MY_NMF_new_data =  purity_score(NMF_TP_points, n_cluster)
            # RI = metrics.rand_score(true_class, pred_class)
            # NMI = metrics.cluster.normalized_mutual_info_score()
            # if purity_of_GRass_MY_NMF_new_data > max1 and   RI > max_RI :
            #     al1 = alpha
            #     be1 = beta
            #     max1 = purity_of_GRass_MY_NMF_new_data
            #     max_RI = RI
        #   kmean on NMF result
            kmean_result = skcl.k_means(X=NMF_result, n_clusters=n_cluster)
            end_time=time.time()
            eval(true_class,kmean_result[1])


def eval(true_class, pred_class):
    contingency_matrix = metrics.cluster.contingency_matrix(true_class, pred_class )
    n = sum(sum(contingency_matrix))
    c2 = contingency_matrix ** 2

    tt = np.sum(contingency_matrix, 1)
    nis = sum(tt ** 2)      # sum of squares of sums of rows
    tt = np.sum(contingency_matrix, 0)
    njs = sum(tt ** 2)      # sum of squares of sums of columns
    TP = 0.5 * sum(sum(c2 - contingency_matrix))
    FP = 0.5 * (njs - n) - TP
    FN = 0.5 * (nis - n) - TP
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F_score = 2 * P * R / (P + R)
#    print("%f old" %F_score)
    ps = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    RI_score = metrics.rand_score(true_class, pred_class)
    NMI_score = normalized_mutual_info_score(true_class, pred_class)
    result[row][4 * kk] = ps
    result[row][4 * kk + 1] = NMI_score
    result[row][4 * kk + 2] = RI_score
    result[row][4 * kk + 3] = F_score
def LNMFS(data , n_cluster , diag , weight ,true_class ):
            global kk, row,result, alpha , beta , end_time
            NMF_result = NMF_decompose_lnmfs(data, n_cluster, alpha=alpha, beta=beta, landa=0 ,diag=diag , weight=weight )
            #   without kmean
            # pred_class = NMF_result.argmax(axis=1)
            # NMF_TP_points = np.column_stack((true_class, pred_class))
            # purity_of_LNMFS_org_data =  purity_score(NMF_TP_points, n_cluster)
            # RI = metrics.rand_score(true_class, pred_class)
            # if purity_of_LNMFS_org_data > max1:
            #     al1 = alpha
            #     be1 = beta
            #     max1 = purity_of_LNMFS_org_data
            #     max_RI = RI
            #   kmean on NMF result
            kmean_result = skcl.k_means(X=NMF_result, n_clusters=n_cluster)
            end_time=time.time()
            eval(true_class,kmean_result[1])
def show_result(data):
    for kk in range(0,4):
        for jj in range(0,12):
            print("%f / %f " %(data[max_iter+0][4*jj+kk] , data[max_iter+1][4*jj+kk] ) ) #average and standard dev
        print("  ")
######################################################################## initialization
start_time=end_time=0
Grass_max_iter = 500
max_iter= 10
NMF_type =1 #    0:LNMFS   1:MY
result = np.zeros(shape=[max_iter+2, 56])
time_criteria = np.zeros(max_iter)

for kk in range (3,4):
# kk  0 :coil100   1:breastw  2:pendigit  3:yale 4:optdigit   5:coil20  # 6:ORL  7:yeast 8:ecoli 9:Isolet  10:CNAE 11:USPS  12:MNIST  13:mpeg7
    org_data , true_class , n_cluster , n_sample , n_neighbors = load_data(kk)
    for row in range (0,max_iter):
        if NMF_type ==0 :
            alpha = 1
            beta = 316.227766  # 10^2.5
            start_time=time.time()
            weight = KNN.kneighbors_graph(X=org_data, n_neighbors=n_neighbors).toarray()
            diag = weight.sum(axis=0)
            diag = np.diagflat([diag])
            print("LNMFS_on_stifel_manifold_on_org_data")
            LNMFS(org_data ,n_cluster , diag , weight ,true_class )
            time_criteria[row] = end_time - start_time
        else :
            alpha = 0.1  # best for my algorithm
            beta = 3162.27766  # 10^3.5
            start_time=time.time()
            weight = KNN.kneighbors_graph(X=org_data, n_neighbors=n_neighbors).toarray()
            lap_mat = csgraph.laplacian(weight , symmetrized=True )
            new_data = stiefel_manifold(weight, lap_mat, n_cluster, Grass_max_iter, teta=0)
            Weight = KNN.kneighbors_graph(X=new_data, n_neighbors = n_neighbors).toarray()
            diag = Weight.sum(axis=0)
            diag = np.diagflat([diag])
            print("stifel_manifol_my_nmf_on_new_data %f"% row)
            MY_decomp(new_data ,n_cluster ,true_class , diag , weight )
            time_criteria[row] = end_time - start_time

result[max_iter,:]= np.average(result[0:max_iter,:],axis=0)
result[max_iter+1,:]= np.std(result[0:max_iter,:],axis=0)
print(time_criteria)
print(result)
# show_result(result)
exit(0)
