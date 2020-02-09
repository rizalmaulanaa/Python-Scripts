# # CLARA

import numpy as np
import pandas as pd
import random as rd

class CLARA(object):
    
    def __init__(self,n_clusters=2,max_iter=5):
        self.k = n_clusters
        self.max_iter = max_iter
        self.cost, self.medoid, self.cluster = [], [], []
        
    def fit(self,df,status=True):
        if status:
            self.df = df
            print('CLARA with {} clusters done'.format(self.k))
        cost, medoid, cluster = self.k_medoid(df)
        return cost, medoid, cluster
        
    def transform(self,df_sisa):
        cost, cluster = self.total_cost(self.medoid,df_sisa)
        for i,j in zip(df_sisa.index,cluster):
            self.labels_fit[i] = j
            
        labels = {k:v for k,v in zip(sorted(self.labels_fit),self.labels_fit.values())}
        labels = list(labels.values())
        return labels
        
    def fit_transform(self,df):
        self.df = df
        subsam = []
        for loop in range(self.max_iter):
            subsample = rd.sample([i for i in df.index], (40+self.k*2))
            df_subsample = df.iloc[subsample,:]
            cost, medoid, cluster = self.fit(df_subsample,status=False)
            subsam.append(subsample)
            self.cost.append(cost)
            self.medoid.append(medoid)
            self.cluster.append(cluster)
        
        index = self.cost.index(min(self.cost))
        self.cost = self.cost[index]
        self.medoid = self.medoid[index]
        self.cluster = self.cluster[index]
        self.labels_fit = {i:j for i,j in zip(subsam[index],self.cluster)}
        df_sisa = df.drop(subsam[index])
        labels = self.transform(df_sisa)
        return labels
    
    def k_medoid(self,df_subsample):
        cost_list, medoid_list, cluster_list = [], [], []
        loop = True
        while loop:
            if len(cost_list) == 1 and len(medoid_list) == 1:
                num = 1
            else:
                num = 2

            for k in range(0,num):
                medoid = rd.sample([i for i in df_subsample.index],self.k)
                medoid_list.append(medoid)
                cost, cluster = self.total_cost(medoid,df_subsample)
                cost_list.append(cost)
                cluster_list.append(cluster)

            index = cost_list.index(min(cost_list))
            cost_list = [cost_list[index]]
            medoid_list = [medoid_list[index]]
            cluster_list = [cluster_list[index]]
            if index == 0 : 
                loop = False
        
        return cost_list[0], medoid_list[0], cluster_list[0]
        
    def total_cost(self,medoid,df):
        cost = 0
        cluster = []
        
        for i in df.index:
            temp = []
            for j in medoid:
                temp.append(self.distance(df.loc[i],self.df.loc[j]))
            
            index = temp.index(min(temp))
            cluster.append(index)
            cost += temp[index]
            
        return cost,cluster
    
    def distance(self,vector1,vector2):
        total = 0
        for a1,a2 in zip(vector1,vector2):
            total += np.power(a1-a2,2)
            
        return np.sqrt(total)


# # Documentations

# Function:

# __init__                     : inisialisasi jumlah cluster yang diinginkan dan maksimum iterasi
# fit                          : melakukan k_medoid untuk data awal (data subsample)
# transform                    : melakukan perhitungan jarak terdekat dengan medoid
# fit_transform                : melakukan algoritme CLARA secara langsung
# k_medoid                     : melakukan k_medoid terhadap data yang ada
# total_cost                   : menghitung cost dan melakukan mengelompokan data
# distance                     : perhitungan jarak menggunakan euclidean distance

# Variables:

# self.k                       : menyimpan jumlah cluster yang akan dibuat
# self.max_iter                : menyimpan maksimum dari iterasi CLARA
# self.df                      : menyimpan data yang ingin di clustering
# self.cost                    : menyimpan nilai cost final
# self.medoid                  : menyimpan nilai medoid final
# self.labels_fit              : menyimpan hasil clustering dari method fit
# cluster                      : menyimpan hasil clustering (label)
# cost                         : menyimpan nilai cost sementara
# labels                       : menyimpan nilai label
# subsam                       : menyimpan index subsample keseluruhan
# subsample                    : menyimpan index yang menjadi subsample
# df_subsample                 : menyimpan data subsample
# df_sisa                      : menyimpan data selain dari data subsample
# cost_list                    : menyimpan nilai cost untuk dibandingkan
# medoid_list                  : menyimpan nilai medoid untuk dibandingkan bedasarkan cost
# cluster_list                 : menyimpan nilai cluster untuk dibandingkan bedasarkan cost
# loop                         : menandakan looping ke berapa
# medoid                       : menyimpan nilai medoid sementara
# index                        : menyimpan index dari nilai cost terkecil
# temp                         : menyimpan sebuah nilai sementara
# total                        : menyimpan nilai jarak