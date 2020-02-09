import pandas as pd
import numpy as np

class Parsing(object):
    
    def __init__(self,types,chunksize=5000000):
        self.chunksize = chunksize
        self.types = types
    
    def text_to_feather(self,path_in=[],path_out=[],sep='\t',skiprows=4):
        k = 0
        for in_,out_ in zip(path_in,path_out):
            df = []
            print('Load data {}...'.format(k+1))
            for i in pd.read_csv(in_, sep=sep, dtype=self.types, skiprows=skiprows, chunksize=self.chunksize):
                df.append(i)
                
            train_df = pd.concat(df)
            del df
            train_df=train_df.drop(train_df.tail(1).index)
            print(train_df.info())
            train_df.to_feather(out_)
            print('Finish parsing data {}...'.format(k+1))
            k += 1

# Variables

# self.chunksize               : menentukan berapa banyak chunk yang akan digunakan
# self.types                   : menentukan tipe dari kolom yang ada
# path_in                      : menentukan file path yang akan digunakan (.txt)
# path_out                     : menentukan file path yang akan disimpan (.feather)
# sep                          : pemisah pada data yang akan diparsing
# skiprows                     : melewat berapa baris pada data yang akan diparsing
# k                            : menentukan jumlah looping yang sudah dijalankan
# df                           : memasukan data sebanyak self.chunksize
# train_df                     : membuat dataframe dengan menggabungkan df

# Methods

# __init__                     : menetapkan data yang merupakan global variable
# text_to_feather              : melakukan parsing dari file .txt menjadi .feather