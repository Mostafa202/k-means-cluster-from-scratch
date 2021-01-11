import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import sys


dataset=pd.read_csv('Mall_Customers.csv')

data=dataset.iloc[:,1:].values

from sklearn.preprocessing import *

lb=LabelEncoder()
data[:,0]=lb.fit_transform(data[:,0])



def find_col_min_max(items):
    minima=np.min(items,axis=0)
    maxima=np.max(items,axis=0)
    return minima,maxima



def initialize_centroids(data,k,cmin,cmax):
    means=[[0 for i in range(len(data[0]))] for j in range(k)]
    for mean in means:
        for i in range(len(mean)):
            mean[i]=np.random.uniform(cmin[i]+1,cmax[i]-1)
    return means


m=initialize_centroids(data,2,find_col_min_max(data)[0],find_col_min_max(data)[1])

def calc_distance(dist1,dist2):
    d1=np.array(dist1)
    d2=np.array(dist2)
    return np.sqrt(np.sum((d1-d2)**2))

def update_mean(n,mean,item):
    for i in range(len(mean)):
        m=mean[i]
        m=(m*(n-1)+item[i])/n
        mean[i]=round(m,3)
    return mean

def classify(means,item):
    minimum=sys.maxsize
    index=-1
    for i in range(len(means)):
        dis=calc_distance(item,means[i])
        if dis < minimum:
            minimum=dis
            index=i
            
    return index
        

def calc_means(k,items,max_iterations=100000):
    cmin,cmax=find_col_min_max(items)
    means=initialize_centroids(items,k,cmin,cmax)
    cluster_size=[0 for j in range(k)]
    belongs_to=[0 for j in range(len(items))]
    for e in range(max_iterations):
        no_change=True
        for i in range(len(items)):
            index=classify(means,items[i])
            cluster_size[index]+=1
            csize=cluster_size[index]
            means[index]=update_mean(csize,means[index],items[i])
            if index!=belongs_to[i]:
                no_change=False
                belongs_to[i]=index
        if no_change:
            break
    return means

def find_clusters(means,items):
    clusters=[[] for i in range(len(means))]
    for i in range(len(items)):        
        index=classify(means,items[i])
        clusters[index].append(items[i])
    return clusters

means=calc_means(2,data)

clusters=find_clusters(means,data)

    
    




