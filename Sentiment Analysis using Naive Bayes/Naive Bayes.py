import numpy as np
import pandas as pd
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop = open('stopword tala.txt','r')
stoplist = stop.read().split('\n')
stop.close()
path = '/Users/Rizal Maulana/Downloads/semester 7/TextMin/'


# # Read Data

data = pd.read_excel(path+'Dataset Kelompok 4.xlsx')
korpus = data['Komentar']
label = data['Hasil akhir']
print(korpus)


# # K-Fold

k_fold = []
k = 3
flag = [0,15]

for i in range(0,5):
    temp = []
    for j in flag:
        temp += list(label[j:j+k].index)
    k_fold.append(temp)
    flag = [h+k for h in flag]
    
print(k_fold)


# # Text Preprocessing

full_term = []
for doc in korpus:
    token = re.findall(r'\b[A-Za-z]{1,}',doc)
    term = []
    for word in token:
        if word not in stoplist:
            term.append(stemmer.stem(word.lower()))
    full_term.append(term)

print(full_term)


# # Pembobotan Term

def pembobotan(doc_term,doc_test):
    term = np.unique(sum(doc_term,[]))
    columns = [i for i in range(len(doc_term))]
    dtm = pd.DataFrame(0,index=term,columns=columns)

    for i,doc in enumerate(doc_term):
        for term in doc:
            dtm.loc[term,i] += 1
    
    term_test = np.unique(sum(doc_test,[]))
    columns = [i for i in range(len(doc_test))]
    dtm_test = pd.DataFrame(0,index=dtm.index,columns=columns)
    
    for i,doc in enumerate(doc_test):
        for term in doc:
            if term in dtm.index:
                dtm_test.loc[term,i] += 1
            
    return dtm, dtm_test


# # Naive Bayes

class Naive_Bayes(object):
    
    def train(self,df_train,y_train):
        self.prior = {j:(i/len(y_train)) for i,j in zip(y_train.value_counts(),y_train.value_counts().index)}
        self.cp = pd.DataFrame(0,dtm_train.index,columns=self.prior)
        y_group = y_train.groupby(y_train)
        
        for i in self.prior.keys():
            for j in df_train.index:
                atas = df_train.loc[j,y_group.get_group(i).index].sum()+1
                bawah = df_train[y_group.get_group(i).index].sum().sum()+len(df_train.index)
                self.cp.loc[j,i] = atas/bawah
        
        print('Training data finish')

    def test(self,df_test):
        hasil = []
        
        for i in df_test.columns:
            temp = {}
            for j in self.prior.keys():
                temp[j] = np.prod(self.cp[df_test[i] > 0][j])*self.prior[j]

            hasil.append(max(temp,key=temp.get))
        
        return hasil
    
    def akurasi(self,y_true,y_predict):
        total = 0
        for i,j in zip(y_true,y_predict):
            if i == j:
                total += 1
                
        return total/len(y_true)


# # Evaluasi

akurasi = []
for i in k_fold:
    y_train = label[~label.index.isin(i)]
    y_test = label[i]
    term_train = [full_term[j] for j in range(len(full_term)) if j not in i]
    term_test = [full_term[j] for j in i]

    dtm_train,dtm_test = pembobotan(term_train,term_test)
    dtm_train.columns = y_train.index
    dtm_test.columns = y_test.index

    nb = Naive_Bayes()
    nb.train(dtm_train,y_train)
    y_predict = nb.test(dtm_test)
    print(y_test)
    print(y_predict)
    akurasi.append(nb.akurasi(y_test,y_predict))

print(np.mean(akurasi))


# # Data Latih

print([full_term[j] for j in range(len(full_term)) if j not in k_fold[0]])


# # Data Uji

print([full_term[i] for i in k_fold[0]])