import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

class Segmentation(object):

    def __init__(self,limit=None):
        self.limit = limit
        self.timeseries = []
        self.df_preprocess = []
        self.rfm = []
        
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder())
        ])

        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(missing_values = 0, strategy='median')),
            ('scaler', MinMaxScaler())
        ])
        
    def preprocessing(self,list_file,top_number=None,limit_pca=50,filter_cat=[],filter_app=[]):
        for loop,file in enumerate(list_file):
            
#           read data  
            print('Load data {}....'.format(loop+1))
            df = pd.read_feather(file)
            print('Finish load data {}'.format(loop+1))
            
#           filtering
            df_ac = df[df['indi_apps_201904.activity_sec']>=600]
            df_ac = df_ac[~df_ac['indi_apps_201904.category_name'].isin(filter_cat)]
            df_ac = df_ac[~df_ac['indi_apps_201904.application_name'].isin(filter_app)]
            
#           take top number in application columns
            top = df_ac['indi_apps_201904.application_name'].value_counts().head(top_number).index
            df_ac = df_ac[df_ac['indi_apps_201904.application_name'].isin(top)]
            df_ac.index = range(len(df_ac))
            
#           select the limit
            df_ac = df_ac.head(self.limit)
            df_ac['indi_apps_201904.ds'] = pd.to_datetime(df['indi_apps_201904.ds'].astype(str),format='%Y%m%d')
            self.df_preprocess.append(df_ac.drop(columns=['indi_apps_201904.activity_sec',
                                                          'indi_apps_201904.volume_in','indi_apps_201904.volume_out']))
            df_ac = df_ac.drop(columns=['indi_apps_201904.user_id', 'indi_apps_201904.ds'])
            print('Finish filtering data {}'.format(loop+1))
            
#           pipeline
            categorical_columns = df_ac.select_dtypes(include='object').columns
            numeric_columns = df_ac.select_dtypes(include='float64').columns

            preprocessor = ColumnTransformer(transformers=[
                ('cat', self.categorical_transformer, categorical_columns),
                ('num', self.numeric_transformer, numeric_columns)
            ])
            
            df_temp = preprocessor.fit_transform(df_ac)
            df_norm = pd.DataFrame(df_temp.toarray())
            print('Finish normalization data {}'.format(loop+1))
            
#           pca
            pca = PCA(n_components=limit_pca).fit(df_norm)
            print('Percentage of variance : {}'.format(sum(pca.explained_variance_ratio_)))
            pca = pca.transform(df_norm)
            columns_pca = ["PC "+str(i) for i in range(len(pca[0]))]
            df_pca = pd.DataFrame(pca, index=range(len(df_norm)), columns=columns_pca)
            print('Finish PCA data {}'.format(loop+1))
            
            self.timeseries.append(df_pca)
            del df,df_ac,top,categorical_columns,numeric_columns,df_norm,pca,columns_pca,df_pca
            print('Finish preprocessing data {}'.format(loop+1))
        
        return self.timeseries

    def RFM_preprocessing(self,list_file,top_number=None,filter_cat=[],filter_app=[]):
        for loop,file in enumerate(list_file):
            
#           read data  
            print('Load data {}....'.format(loop+1))
            df = pd.read_feather(file)
            print('Finish load data {}'.format(loop+1))
            
#           filtering
            df_ac = df[df['indi_apps_201904.activity_sec']>=600]
            df_ac = df_ac[~df_ac['indi_apps_201904.category_name'].isin(filter_cat)]
            df_ac = df_ac[~df_ac['indi_apps_201904.application_name'].isin(filter_app)]
            
#           take top number in application columns
            top = df_ac['indi_apps_201904.application_name'].value_counts().head(top_number).index
            df_ac = df_ac[df_ac['indi_apps_201904.application_name'].isin(top)]
            df_ac.index = range(len(df_ac))
            
#           select the limit
            df_ac = df_ac.head(self.limit)
            df_ac['indi_apps_201904.ds'] = pd.to_datetime(df['indi_apps_201904.ds'].astype(str),format='%Y%m%d')
        
            self.rfm.append(df_ac)
            del df,df_ac,top
            print('Finish filtering data {}'.format(loop+1))
        
        return self.rfm
        
    def load_top(self,df,name='app',top_number=None):
        if name == 'app':
            return df['indi_apps_201904.application_name'].value_counts().head(top_number).index
        elif name == 'cat':
            return df['indi_apps_201904.category_name'].value_counts().head(top_number).index
    
    
# Variables

# self.limit                   : menentukan berapa banyak jumlah data yang diambil
# self.timeseries              : menyimpan semua data yang sudah di-preprocessing
# self.df_preprocess           : menyimpan data selain tipe numerik
# self.rfm                     : menyimpan data hasil preprocessing RFM
# self.categorical_transformer : pipeline untuk merubah kolom kategori
# self.numeric_transformer     : pipeline untuk merubah kolom numerik
# list_file                    : menyimpan path untuk file .feather yang akan dibuka
# top_number (preprocessing)   : menentukan berapa banyak nilai yang diambil bedasarkan appication name
# limit_pca                    : menentukan berapa banyak fitur yang akan diambil
# filter_cat                   : menyimpan data yang akan dihapus bedasarkan category name
# filter_app                   : menyimpan data yang akan dihapus bedasarkan application name
# df (preprocessing)           : menyimpan data lengkap
# df_ac                        : menyimpan data yang sudah di filtering
# top                          : menyimpan data yang selain dari filter_cat dan filter_app
# categorical_columns          : menyimpan kolom kategori
# numeric_columns              : menyimpan kolom numerik
# preprocessor                 : pipeline untuk merubah semua data menjadi numerik
# df_temp                      : menjalankan pipeline dengan df_ac
# df_norm                      : menyimpan hasil preprocessor kedalam dataframe
# pca                          : menjalankan pca dengan df_norm
# columns_pca                  : membuat kolom sebanyak jumlah pc yang dibuat
# df_pca                       : menyimpan pca kedalam dataframe
# df (load_top)                : menyimpan dataframe yang akan diproses
# name                         : menentukan akan melakukan bedasarkan application name atau category name
# top_number (load_top)        : menentukan berapa banyak data yang akan ditampilkan


# Methods

# __init__                     : menetapkan data yang merupakan global variable
# preprocessing                : melakukan preprocessing bedasarkan list_file yang ada, dengan output semua data preprocessing
# RFM_preprocessing            : melakukan filtering untuk penggunaan RFM
# load_top                     : menampilkan jumlah data terbanyan