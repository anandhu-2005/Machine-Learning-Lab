import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("digits_dataset.csv")
X=data.drop(columns=["label"]).values
def dbscan(X,eps,minpts):
    n=len(X)
    labels=np.zeros(n,dtype=int)
    cluster=0
    for i in range(n):
        if labels[i]!=0:
            continue
        neighbors=[]
        for j in range(n):
            d=np.sqrt(np.sum((X[i]-X[j])**2))
            if d<=eps:
                neighbors.append(j)
        if len(neighbors)<minpts:
            labels[i]=-1
            continue
        cluster+=1
        labels[i]=cluster
        k=0
        neighbors_set=set(neighbors)
        while k<len(neighbors):
            point=neighbors[k]
            if labels[point]==-1:
                labels[point]=cluster
            if labels[point]!=0:
                k+=1
                continue
            labels[point]=cluster
            new_neighbors=[]
            for j in range(n):
                d=np.sqrt(np.sum((X[point]-X[j])**2))
                if d<=eps:
                    new_neighbors.append(j)
            if len(new_neighbors)>=minpts:
                for p in new_neighbors:
                    if p not in neighbors_set:
                        neighbors.append(p)
                        neighbors_set.add(p)
            k+=1
    return labels
cluster_points=[]
noise_points=[]
eps=[5,10,20,21,22,23,24,25,26]
for e in eps:
    labels=dbscan(X,e,5)
    clus=len(set(labels)-{-1})
    cluster_points.append(clus)
    noi=np.sum(labels==-1)
    noise_points.append(noi)
    print(e,"\t",clus,"\t",noi)
plt.plot(eps,cluster_points,marker="o",label="epsVScluster")
plt.plot(eps,noise_points,marker="o",label="epsvsnoise")
plt.legend()
plt.show()
    
