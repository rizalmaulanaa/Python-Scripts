import pandas as pd
import numpy as np
import os
import parsing as pars
import preprocessing_segmentation as pg
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
get_ipython().run_line_magic('matplotlib', 'inline')

mentari_path = '/Users/ASUS/Documents/Python Scripts/'
rizal_path = '/Users/Rizal Maulana/Downloads/semester 7/PKL/datas/'


# # Parsing .txt into .feather

# In[2]:


# traintypes = {'indi_apps_201904.user_id': 'object',
#               'indi_apps_201904.category_name': 'object', 
#               'indi_apps_201904.application_name': 'object',
#               'indi_apps_201904.activity_sec': 'float64',
#               'indi_apps_201904.volume_in': 'float64',
#               'indi_apps_201904.volume_out': 'float64',
#               'indi_apps_201904.ds': 'object'}

# path_in = [rizal_path+i for i in os.listdir(rizal_path) if i.endswith('.txt')]
# path_out = [i.split('.')[0]+'.feather' for i in path_in]


# In[3]:


# pr = pars.Parsing(traintypes,chunksize=5000000)
# pr.text_to_feather(path_in=path_in,path_out=path_out,sep=';')


# # Read data

# In[4]:


list_file = [rizal_path+i for i in os.listdir(rizal_path) if i.endswith('.feather')]
print(list_file)


# # Analysis data

# In[5]:


segment = pg.Segmentation(limit=200000)
df = pd.read_feather(list_file[0])


# In[6]:


print('Users : {:,}'.format(len(np.unique(df['indi_apps_201904.user_id']))))
print('Length of unique columns application name : {:,}'.format(len(np.unique(df['indi_apps_201904.application_name']))))
print('Length of unique columns category name : {:,}'.format(len(np.unique(df['indi_apps_201904.category_name']))))


# In[7]:


print('Top value by application name :')
print(segment.load_top(df,name='app',top_number=100))


# In[8]:


print('Top value by category name :')
print(segment.load_top(df,name='cat'))


# # Filtering

# In[9]:


list_del_cat = ['Ads','Attack','CloudStorage','Content','MDM','Misc','Proxy','Remote','Sync',
                'Tracker','Update','Weather','non-established','unclassified','Browser','DeviceServices',
                'File','FileSharing','GoogleServices','Media','RPC','Security','VPN']
list_del_app = ['GooglePortalDetection','TouchPal','CleanMaster','BBM-UCI', 'AdultSites','SinaCN']


# # Preprocessing

# In[10]:


print('Total datas : {}\n'.format(len(list_file)))
timeseries = segment.preprocessing(list_file,top_number=100,limit_pca=60s,
                                   filter_cat=list_del_cat,filter_app=list_del_app)
df_ = segment.df_preprocess


# # Optimal K-means

# In[16]:


def optimal(df,awal,akhir,sample):
    eval_,cluster = [],[]
    for i in range(awal,akhir):
        labels = KMeans(n_clusters=i).fit_predict(df)
        sil = round(silhouette_score(df, labels, sample_size=sample), 3)
        eval_.append(sil)
        cluster.append(i)
        print('{} Cluster, Silhouette Score {}'.format(i,sil))
        
    return eval_,cluster

tm = pd.concat(timeseries)
tm.index = range(len(tm))


# # Evaluation

# ## Silhouette Score

# In[19]:


sample = 20000
ss,cluster = optimal(tm,2,10,sample)
plt.figure(figsize=(16,7))
sns.pointplot(x=cluster, y=ss)
plt.title('Silhouette Score with {:,} datas or sample'.format(sample))
plt.tight_layout()
plt.xlabel('Cluster')
plt.ylabel('Silhouette Score')
plt.show()


# # KMeans

# In[20]:


df_tm = pd.concat(df_)
labels = KMeans(n_clusters=9).fit_predict(tm)
df_tm['Cluster'] = labels
print(df_tm)


# # Group by Cluster

# In[21]:


df_cluster = df_tm.groupby('Cluster')
df_cluster.get_group(8)


# # Summary every Segment

# In[26]:


for i in np.unique(labels):
    print('Cluster {}'.format(i))
    print('Unique category name : {}'.format(list(np.unique(df_cluster.get_group(i)['indi_apps_201904.category_name']))))
    print('Unique application name : {}\n'.format(
        list(np.unique(df_cluster.get_group(i)['indi_apps_201904.application_name']))))


# # Documentations
# Variables

# mentari_path                 : path file pada laptop mentari
# rizal_path                   : path file pada laptop rizal
# traintypes                   : tipe kolom pada data .txt
# path_in                      : path file yang akan diparsing
# path_out                     : path file yang akan disimpan
# pr                           : inisialisasi class Parsing
# list_file                    : menyimpan path .feather yang akan diproses
# segment                      : inisialisasi class preprocessing_segment
# df                           : membuka file .feather
# segment.load_top             : menampilkan nilai teratas bedasarkan application / category name
# list_del_cat                 : menyimpan daftar yang akan dihapus bedasarkan categoty name
# list_del_app                 : menyimpan daftar yang akan dihapus bedasarkan application name
# timeseries                   : menyimpan semua dataframe yang sudah di preprocessing
# df_                          : menyimpan data selain numerik
# tm                           : menyimpan data timeseries menjadi 1
# sample                       : menentukan berapa banyak sample yang digunakan untuk proses silhouette score
# df_tm                        : menyimpan df_ menjadi 1
# labels                       : menyimpan hasil clustering
# df_cluster                   : melakukan grouping sesuai dengan clustering

# Method

# optimal                      : menentukan cluster yang optimal