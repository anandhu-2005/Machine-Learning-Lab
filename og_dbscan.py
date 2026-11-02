import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("digits_dataset.csv")
X=data.drop(columns=["label"]).values
n=len(X)
def dbscan(X,eps,minpts):
    labels=np.zeros(n,dtype=int)
    cluster=0
    for i in range(n):
        if labels[i]!=0:
            continue
        neighbors=[]
        for j in range(n):
            d=np.sqrt(np.sum((X[j]-X[i])**2))
            if d<=eps:
                neighbors.append(j)
        if len(neighbors) < minpts:
            labels[i]=-1
            continue
        cluster+=1
        labels[i]=cluster
        k=0
        while(k<len(neighbors)):
            point=neighbors[k]
            if labels[point]==-1:
                labels[point]=cluster
            if labels[point]!=0:
                k+=1
                continue
            labels[point]=cluster
            new_neighbors=[]
            for j in range(len(X)):
                d=np.sqrt(np.sum((X[point]-X[j])**2))
                if d <= eps:
                    new_neighbors.append(j)
            if len(new_neighbors)>=minpts:
                neighbors.extend(new_neighbors)
            k+=1
    return labels
clusters_point=[]
noise=[]
e = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
for eps in e:
    labels=dbscan(X,eps,5)
    clus=len(set(labels)-{-1})
    clusters_point.append(clus)
    noi=np.sum(labels==-1)
    noise.append(noi)
    print("Eps:", eps, "Clusters:", clus, "Noise:", noi)
plt.plot(e,clusters_point,label="epsvscluster",marker="o")
plt.grid()
plt.legend()
plt.xlabel("eps")
plt.ylabel("no of clusters")
plt.show()
plt.plot(e,noise,label="epsvsnoise",marker="o",color="red")
plt.grid()
plt.legend()
plt.xlabel("eps")
plt.ylabel("noise")
plt.show()
    
    
