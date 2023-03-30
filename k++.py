import matplotlib.pyplot as plt 

from sklearn          import metrics
from sklearn.cluster  import KMeans
from sklearn.datasets import make_blobs


centers = [(5, 5), (6, 6), (7, 3)]
NUM_CLUSTER = len(centers)
COUNT_DATA = 30

X, y = make_blobs(n_samples=COUNT_DATA, centers=centers, random_state=50)

plt.scatter(X[:,0],X[:,1], label='True Position')

kmeans = KMeans(n_clusters=NUM_CLUSTER) 
kmeans.fit(X)

plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow') 
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')

#===================================================================#
print('Data:')
count = 1
for ans, pos in zip(y, X):
    print(f'{count})\t({pos[0]},\t{pos[1]}):\t{ans}')
    count+=1

print('\nStarting position:')
count = 1
for pos in kmeans.cluster_centers_:
    print(f'{count}) x: {pos[0]}\n   {pos[1]}\n')
    count+=1

print('ARI: ', metrics.adjusted_rand_score(y, kmeans.labels_))
print('AML: ', metrics.adjusted_mutual_info_score(y, kmeans.labels_))
#===================================================================#

plt.show()














# import matplotlib.pyplot as plt 
# import numpy as np 
# import random
# from sklearn.cluster import KMeans
# from sklearn import metrics

# NUM_CLUSTER = 3
# COUNT_DATA = 30

# X = np.random.randint(0, 90, (COUNT_DATA, 2))

# plt.scatter(X[:,0],X[:,1], label='True Position')

# kmeans = KMeans(n_clusters=NUM_CLUSTER) 
# kmeans.fit(X)

# plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow') 
# plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')

# #===============================================#
# print('Data:')
# count = 1
# for pos in X:
#     print(f'{count}) x:', pos[0], '\ty:', pos[1])
#     count+=1

# print('\nStarting position:')
# count = 1
# for pos in kmeans.cluster_centers_:
#     print(f'{count}) x: {pos[0]}\n   {pos[1]}\n')
#     count+=1

# y = np.array([random.randint(0, NUM_CLUSTER-1) for i in range(COUNT_DATA)])
# # print(y)
# # print(kmeans.labels_)
# print('ARI: ', metrics.adjusted_rand_score(y, kmeans.labels_))
# print('AML: ', metrics.adjusted_mutual_info_score(y, kmeans.labels_))
# #===============================================#

# plt.show();