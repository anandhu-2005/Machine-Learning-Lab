import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

X = data[["Annual Income (k$)", "Spending Score (1-100)"]].values
l=0

# K-Means Clustering
def kmeans(X, k):
    np.random.seed(42)
    centers = X[np.random.choice(len(X), k, replace=False)]
    l=len(centers)
    for _ in range(100):
        labels = []

        # Assign points to nearest center
        for point in X:
            distances = []

            for center in centers:
                d = np.sqrt(np.sum((point - center) ** 2))
                distances.append(d)

            labels.append(np.argmin(distances))

        labels = np.array(labels)

        # Update centers
        new_centers = []

        for i in range(k):
            cluster = X[labels == i]

            if len(cluster) > 0:
                new_centers.append(np.mean(cluster, axis=0))
            else:
                new_centers.append(centers[i])

        centers = np.array(new_centers)

    return labels


# Hierarchical Clustering

def hierarchical(X, k):
    clusters = [[i] for i in range(len(X))]

    while len(clusters) > k:
        min_dist = float("inf")

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                max_dist = 0

                for a in clusters[i]:
                    for b in clusters[j]:
                        d = np.sqrt(np.sum((X[a] - X[b]) ** 2))
                        max_dist = max(max_dist, d)

                if max_dist < min_dist:
                    min_dist = max_dist
                    merge_i = i
                    merge_j = j

        clusters[merge_i].extend(clusters[merge_j])
        del clusters[merge_j]

    labels = np.zeros(len(X), dtype=int)

    for i in range(len(clusters)):
        for point in clusters[i]:
            labels[point] = i

    return labels


# Run both algorithms
k = 5

km = kmeans(X, k)
hc = hierarchical(X, k)


# Display cluster counts
print("K-Means Cluster Counts:")
for i in range(k):
    print("Cluster", i + 1, ":", np.sum(km == i))

print("\nHierarchical Cluster Counts:")
for i in range(k):
    print("Cluster", i + 1, ":", np.sum(hc == i))

print(l)
# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=km, cmap="rainbow")
plt.title("K-Means Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=hc, cmap="rainbow")
plt.title("Hierarchical Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")

plt.tight_layout()
plt.show()
