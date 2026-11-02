
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

# Read dataset
data = pd.read_csv("digits_dataset.csv")

# Separate features and labels
X = data.drop(columns=["label"]).values

def dbscan(X, eps, min_pts):
    labels = np.zeros(len(X), dtype=int)
    cluster = 0

    for i in range(len(X)):
        if labels[i] != 0:
            continue

        neighbors = []

        for j in range(len(X)):
            distance = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            if distance <= eps:
                neighbors.append(j)

        if len(neighbors) < min_pts:
            labels[i] = -1
            continue

        cluster += 1
        labels[i] = cluster
        k = 0

        while k < len(neighbors):
            point = neighbors[k]

            if labels[point] == -1:
                labels[point] = cluster

            if labels[point] != 0:
                k += 1
                continue

            labels[point] = cluster
            new_neighbors = []

            for j in range(len(X)):
                distance = np.sqrt(np.sum((X[point] - X[j]) ** 2))
                if distance <= eps:
                    new_neighbors.append(j)

            if len(new_neighbors) >= min_pts:
                neighbors.extend(new_neighbors)

            k += 1

    return labels


# Analyze different eps values
for eps in [20, 30, 40, 50]:
    labels = dbscan(X, eps, 5)

    clusters = len(set(labels) - {-1})
    noise = np.sum(labels == -1)

    print("Eps:", eps)
    print("Number of clusters:", clusters)
    print("Noise points:", noise)
    print()


# Plot cluster count variation
eps_values = [20, 30, 40, 50]
cluster_counts = []

for eps in eps_values:
    labels = dbscan(X, eps, 5)
    clusters = len(set(labels) - {-1})
    cluster_counts.append(clusters)

plt.plot(eps_values, cluster_counts, marker="o")
plt.xlabel("Eps")
plt.ylabel("Number of Clusters")
plt.title("DBSCAN: Effect of Eps on Cluster Count")
plt.grid()
plt.show()
