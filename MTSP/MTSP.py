import pandas as pd
import numpy as np
import random as rd

path = '/Users/Rizal Maulana/Downloads/semester 7/Alev/'

df_node = pd.read_excel(path+'Data Jarak.xlsx',index_col=0,header=1)
mapping = pd.Series(df_node.index)
df_node.index = range(len(df_node.index))
df_node.columns = range(len(df_node.columns))
print(df_node)


# # M-TSP


class MTSP(object):
    def __init__(self,df_node,awal,flag,populasi=10,generasi=1):
        self.df_node = df_node
        self.awal = awal
        self.flag = flag
        self.populasi = populasi
        self.generasi = generasi
        self.df_populasi = pd.DataFrame(0,index=range(0,populasi),columns=['Kromosome','Cost','Fitness'])
        
    def main(self,cr,mr):
        krom,cost,fitness = [],[],[]
        for loop in range(self.populasi):
            kromosome = rd.sample([i for i in self.df_node.index if i != self.awal],len(self.df_node.index)-1)
            cost_temp = self.total_cost(kromosome)
            fitnes_temp = self.total_fitness(cost_temp)
            krom.append(kromosome)
            cost.append(cost_temp)
            fitness.append(fitnes_temp)

        self.df_populasi['Kromosome'] = krom
        self.df_populasi['Cost'] = cost
        self.df_populasi['Fitness'] = fitness
        
        for loop in range(self.generasi):
            print('Generasi Ke-{}'.format(loop))
            self.reproduction(cr,mr)
            self.selection()
            print('Populasi Bertahan :')
            print(self.df_populasi.head())
#         print(self.df_populasi.head(1)['Fitness'],'\n')

    def reproduction(self,cr,mr):
        self.crossover(cr)
        self.mutation(mr)
        
        krom_ = pd.Series(self.child)
        cost_ = pd.Series([self.total_cost(i) for i in self.child])
        fitness_ = pd.Series([self.total_fitness(i) for i in cost_])
        df_child = pd.concat([krom_,cost_,fitness_],axis=1)
        df_child.columns = ['Kromosome','Cost','Fitness']
        self.df_c = self.df_populasi.copy(deep=True)
        self.df_c = pd.concat([self.df_c,df_child])
        self.df_c.index = range(len(self.df_c))
        
    def crossover(self,cr):
        jml_cros = round(cr*self.populasi)
        self.child = []

        for loop in range(jml_cros):
            cut = rd.randint(self.df_node.index[0],self.df_node.index[-2])
            parent = rd.sample([i for i in self.df_populasi.index],2)
    
            child_temp = [i for k,i in enumerate(self.df_populasi.loc[parent[0],'Kromosome']) if k <= cut]
            for i in self.df_populasi.loc[parent[1],'Kromosome']:
                if i not in child_temp:
                    child_temp.append(i)
            
            self.child.append(child_temp)
       
    def mutation(self,mr):
        jml_mut = round(mr*self.populasi)
        for i in range(jml_mut):
            cut = rd.sample([i for i in self.df_node.index[:-2]],2)
            parent = rd.randint(self.df_populasi.index[0],self.df_populasi.index[-1])

            child_temp = self.df_populasi.loc[parent,'Kromosome']
            child_temp[cut[0]], child_temp[cut[1]] = child_temp[cut[1]], child_temp[cut[0]]
            self.child.append(child_temp)
    
    def selection(self):
        self.df_populasi = self.df_c.nlargest(self.populasi,'Fitness')
        self.df_populasi.index = range(len(self.df_populasi))
        
    def total_cost(self,kromosome):
        jarak = [[self.awal]+kromosome[s:e+1]+[self.awal] for s,e in self.flag]
        tot_jarak = []
        cost = 0

        for i in jarak:
            awal = 0
            temp = []
            for j in i:
                temp.append(i[awal:awal+2])
                awal += 1
            tot_jarak += temp[:-1]

        for i in tot_jarak:
            cost += self.df_node.loc[i[0],i[1]]

        return cost

    def total_fitness(self,cost):
        return 1/cost


# # Pengujian Jumlah cr dan mr

kombinasi = [(0.9,0.1),(0.8,0.2),(0.7,0.3),(0.6,0.4),(0.5,0.5),(0.4,0.6),(0.3,0.7),(0.2,0.8),(0.1,0.9)]
mtsp = MTSP(df_node,0,[(0,7),(8,14)],populasi=10,generasi=20)
for i in kombinasi:
    cr,mr=i
    print('Cr : {}, Mr : {}'.format(cr,mr))
    for loop in range(10):
        print('Percobaan Ke-{}'.format(loop))
        mtsp.main(cr,mr)


# # Pengujian Jumlah Populasi

jenis_populasi = [50,60,70,80,90,100]
for i in jenis_populasi:
    print('Jumlah Populasi : {}'.format(i))
    mtsp = MTSP(df_node,0,[(0,7),(8,14)],populasi=i,generasi=20)
    for loop in range(10):
        print('Percobaan ke-{}'.format(loop))
        mtsp.main(0.9,0.1)


# # Pengujian Jumlah Kurir

salesman = [[(0,14)],[(0,7),(8,14)],[(0,4),(5,9),(10,14)],[(0,3),(4,7),(8,11),(12,14)],[(0,2),(3,5),(6,8),(9,11),(12,14)],
            [(0,1),(2,4),(5,8),(9,10),(11,12),(13,14)],[(0,1),(2,4),(5,6),(7,8),(9,10),(11,12),(13,14)],
            [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,14)],[(0,0),(1,2),(3,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,14)],
            [(0,0),(1,2),(3,3),(4,4),(5,5),(6,7),(8,9),(10,11),(12,13),(14,14)]]
for i in salesman:
    print('Jumlah Salesman : {}'.format(len(i)))
    mtsp = MTSP(df_node,0,i,populasi=70,generasi=20)
    for loop in range(10):
        print('Percobaan ke-{}'.format(loop))
        mtsp.main(0.9,0.1)

        
# # Main

awal = 0
kurir = [(0,14)]

mtsp = MTSP(df_node,awal,kurir,populasi=70,generasi=20)
mtsp.main(0.9,0.1)
df_final = mtsp.df_populasi


# # Mapping


hasil_mapping = []
jarak = [[awal]+df_final.loc[0,'Kromosome'][s:e+1]+[awal] for s,e in kurir]
jarak = [j for i in jarak for j in i]

for i in jarak:
    hasil_mapping.append(mapping[i])
print(hasil_mapping)