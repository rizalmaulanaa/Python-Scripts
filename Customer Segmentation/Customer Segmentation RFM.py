import pandas as pd
import numpy as np
import os
import parsing as pars
import preprocessing_segmentation as pg
import datetime as dt
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

# In[22]:


segment = pg.Segmentation(limit=None)
df = pd.read_feather(list_file[0])


# In[23]:


print('Users : {:,}'.format(len(np.unique(df['indi_apps_201904.user_id']))))
print('Length of unique columns application name : {:,}'.format(len(np.unique(df['indi_apps_201904.application_name']))))
print('Length of unique columns category name : {:,}'.format(len(np.unique(df['indi_apps_201904.category_name']))))


# In[24]:


print('Top value by application name :')
print(segment.load_top(df,name='app',top_number=100))


# In[25]:


print('Top value by category name :')
print(segment.load_top(df,name='cat'))


# # Filtering

# In[9]:


list_del_cat = ['Ads','Attack','CloudStorage','Content','MDM','Misc','Proxy','Remote','Sync',
                'Tracker','Update','Weather','non-established','unclassified','Browser',
                'DeviceServices','File','FileSharing','GoogleServices','Media','RPC','Security','VPN']
list_del_app = ['GooglePortalDetection','TouchPal','CleanMaster','BBM-UCI', 'AdultSites','SinaCN']


# # Preprocessing

# In[10]:


print('Total datas : {}\n'.format(len(list_file)))
df_rfm = segment.RFM_preprocessing(list_file,top_number=100,filter_cat=list_del_cat,filter_app=list_del_app)


# # Recency, Frequency, Monetary (RFM)

# In[11]:


rfm = pd.concat(df_rfm)
date_ = dt.datetime(2019,4,5)

rfm = rfm.groupby('indi_apps_201904.user_id').agg({'indi_apps_201904.ds': lambda x: (date_ - x.max()).days,
                                                   'indi_apps_201904.category_name': lambda x: len(x),
                                                   'indi_apps_201904.activity_sec': lambda x: x.sum()})
rfm['indi_apps_201904.ds'] = rfm['indi_apps_201904.ds'].astype(int)
rfm = rfm.rename(columns={'indi_apps_201904.ds': 'R',
                          'indi_apps_201904.category_name': 'F',
                          'indi_apps_201904.activity_sec': 'M'})
print(rfm)       


# In[12]:


for i in rfm.columns:
    print('{} : min {:,}, max {:,}'.format(i,min(rfm[i]),max(rfm[i])))
    
quan = rfm[['F','M']].quantile(q=[0.25,0.5,0.75])
quan = quan.to_dict()
print(quan)


# In[13]:


def FM(nilai,dict_,kolom):
    if nilai <= dict_[kolom][0.25]:
        return 4
    elif nilai <= dict_[kolom][0.50]:
        return 3
    elif nilai <= dict_[kolom][0.75]: 
        return 2
    else:
        return 1

rfm_segment = rfm.copy().drop(columns=['F','M'])
rfm_segment['F'] = rfm['F'].apply(FM,args=(quan,'F'))
rfm_segment['M'] = rfm['M'].apply(FM,args=(quan,'M'))
print(rfm_segment)


# # Optimal Cluster

# In[14]:


def optimal(df,awal,akhir):
    eval_,cluster = [],[]
    for i in range(awal,akhir):
        labels = KMeans(n_clusters=i).fit_predict(df)
        s = round(silhouette_score(df, labels), 3)
        eval_.append(s)
        cluster.append(i)
        print('{} Cluster, Silhouette Score {}'.format(i,s))
        
    return eval_,cluster


# # Silhouette Score

# In[15]:


ss,cluster = optimal(rfm_segment,2,10)
plt.figure(figsize=(16,7))
sns.pointplot(x=cluster, y=ss)
plt.title('Silhouette Score with {:,} data'.format(len(rfm)))
plt.tight_layout()
plt.xlabel('Cluster')
plt.ylabel('Silhouette Score')
plt.show()


# # Clustering

# In[21]:


labels = KMeans(n_clusters=5).fit_predict(rfm_segment)
rfm_s = rfm_segment.copy()
rfm_s['Segment'] = labels

rfm_group = rfm_s.groupby('Segment')
rfm_group.get_group(4)


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
# df_rfm                       : menyimpan hasil filter dari preprocessing_segment
# rfm                          : menyimpan hasil dari model RFM
# date_                        : inisialisasi tanggal sekarang
# quan                         : membagi data menjadi 4 bagian
# rfm_segment                  : hasil dari pemetaan dari quan
# eval_                        : menyimpan hasil silhouette score
# cluster                      : menyimpan jumlah cluster yang diuji
# labels                       : menyimpan label yang dilakukan oleh kmeans
# ss                           : menyimpan hasil silhouette score
# rfm_s                        : menyimpan hasil cluster
# rfm_group                    : membagi data sesuai dengan cluster

# Method

# optimal                      : menentukan cluster yang optimal
# FM                           : mapping nilai sesuai dengan nilai RFM yang sudah ada