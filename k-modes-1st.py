import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from kmodes.kmodes import KModes
import seaborn as sns

print(f'Reading csv...')
equations = pd.read_csv(r'C:\Users\I502765\OneDrive - Associacao Antonio Vieira\Ciência da Computação\TCC\Practice\data\final-equations.csv') 
print('...Complete')

print(equations.head())
print(equations.columns)
print(equations.describe())
print(equations.info())
print('---------------------------------------------')


col = []
for column in equations.columns:
    if column != 'previousStep':
      col.append(column)
    
equations_nom = equations[col]
print(col)

# Label encoder to categorical data
for col in col:
	le = preprocessing.LabelEncoder()
	equations_nom[col] = le.fit_transform(equations_nom[col])

print(equations_nom.head())

# Determine clusters
cost = []
K = range(1,5)
for num_clusters in list(K):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(equations_nom)
    cost.append(kmode.cost_)
    
plt.plot(K, cost, 'bx-')
plt.xlabel('k clusters')
plt.ylabel('Cost')
plt.title('Elbow Method For Optimal k')
plt.show()

# Clustering
'''
init = Cao
init = Huang
'''
km = KModes(n_clusters=5, init = "Cao", n_init = 1, verbose=1)
cluster_labels = km.fit_predict(equations_nom)
equations['Cluster'] = cluster_labels

kmodes = km.cluster_centroids_
print(kmodes)

shape = kmodes.shape
print(shape)


# Visualize clustering results
for c in col:
  plt.subplots(figsize = (15,5))
  sns.countplot(x='Cluster',hue=c, data = equations)
  plt.show()
  
'''
print("\ncluster 1 observations: ",list(km.labels_).count(0))
print("cluster 2 observations: ",list(km.labels_).count(1))
print("cluster 3 observations: ",list(km.labels_).count(2))
print("cluster 4 observations: ",list(km.labels_).count(3))
print("cluster 5 observations: ",list(km.labels_).count(4))

cluster_1_rows = []
cluster_2_rows = []
cluster_3_rows = []
cluster_4_rows = []
cluster_5_rows = []

for i in range(len(km.labels_)):
    if  km.labels_[i] == 0:
        cluster_1_rows.append(i)
    elif km.labels_[i] == 1:
        cluster_2_rows.append(i)
    elif km.labels_[i] == 2:
        cluster_3_rows.append(i)
    elif km.labels_[i] == 3:
        cluster_4_rows.append(i)
    else:
        cluster_5_rows.append(i)

# creating dataframes from the rows in each cluster
cluster_1_df = pd.DataFrame(equations.iloc[cluster_1_rows], columns=equations.columns)
cluster_2_df = pd.DataFrame(equations.iloc[cluster_2_rows], columns=equations.columns)
cluster_3_df = pd.DataFrame(equations.iloc[cluster_3_rows], columns=equations.columns)
cluster_4_df = pd.DataFrame(equations.iloc[cluster_3_rows], columns=equations.columns)
cluster_5_df = pd.DataFrame(equations.iloc[cluster_3_rows], columns=equations.columns)


# frequency distribution
print("\nCluster 1 left term 1 value distribution:\n", cluster_1_df.leftTerm1.value_counts())
print("\nCluster 1 left operator value distribution:\n", cluster_1_df.leftOperator.value_counts())
print("\nCluster 1 left term 2 value distribution:\n", cluster_1_df.leftTerm2.value_counts())
print("\nCluster 1 right term 1 value distribution:\n", cluster_1_df.rightTerm1.value_counts())
print("\nCluster 1 right operator value distribution:\n", cluster_1_df.rightOperator.value_counts())
print("\nCluster 1 right term 2 value distribution:\n", cluster_1_df.rightTerm2.value_counts())
print('\n---------------------------------------------------------------------------')

print("\nCluster 2 left term 1 value distribution:\n", cluster_2_df.leftTerm1.value_counts())
print("\nCluster 2 left operator value distribution:\n", cluster_2_df.leftOperator.value_counts())
print("\nCluster 2 left term 2 value distribution:\n", cluster_2_df.leftTerm2.value_counts())
print("\nCluster 2 right term 1 value distribution:\n", cluster_2_df.rightTerm1.value_counts())
print("\nCluster 2 right operator value distribution:\n", cluster_2_df.rightOperator.value_counts())
print("\nCluster 2 right term 2 value distribution:\n", cluster_2_df.rightTerm2.value_counts())
print('\n---------------------------------------------------------------------------')

print("\nCluster 3 left term 1 value distribution:\n", cluster_3_df.leftTerm1.value_counts())
print("\nCluster 3 left operator value distribution:\n", cluster_3_df.leftOperator.value_counts())
print("\nCluster 3 left term 2 value distribution:\n", cluster_3_df.leftTerm2.value_counts())
print("\nCluster 3 right term 1 value distribution:\n", cluster_3_df.rightTerm1.value_counts())
print("\nCluster 3 right operator value distribution:\n", cluster_3_df.rightOperator.value_counts())
print("\nCluster 3 right term 2 value distribution:\n", cluster_3_df.rightTerm2.value_counts())
('\n---------------------------------------------------------------------------')

print("\nCluster 4 left term 1 value distribution:\n", cluster_4_df.leftTerm1.value_counts())
print("\nCluster 4 left operator value distribution:\n", cluster_4_df.leftOperator.value_counts())
print("\nCluster 4 left term 2 value distribution:\n", cluster_4_df.leftTerm2.value_counts())
print("\nCluster 4 right term 1 value distribution:\n", cluster_4_df.rightTerm1.value_counts())
print("\nCluster 4 right operator value distribution:\n", cluster_4_df.rightOperator.value_counts())
print("\nCluster 4 right term 2 value distribution:\n", cluster_4_df.rightTerm2.value_counts())
('\n---------------------------------------------------------------------------')

print("\nCluster 5 left term 1 value distribution:\n", cluster_5_df.leftTerm1.value_counts())
print("\nCluster 5 left operator value distribution:\n", cluster_5_df.leftOperator.value_counts())
print("\nCluster 5 left term 2 value distribution:\n", cluster_5_df.leftTerm2.value_counts())
print("\nCluster 5 right term 1 value distribution:\n", cluster_5_df.rightTerm1.value_counts())
print("\nCluster 5 right operator value distribution:\n", cluster_5_df.rightOperator.value_counts())
print("\nCluster 5 right term 2 value distribution:\n", cluster_5_df.rightTerm2.value_counts())

plt.subplots(figsize = (15,5))
sns.countplot(x='Columns', hue=cluster_1_df, data=cluster_1_df)
plt.show()

'''


'''
import numpy as np

for i in range(shape[0]):
    if sum(kmodes[i,:]) == 0:
        print("\ncluster " + str(i) + ": ")
        print("no-skills cluster")
    else:
        print("\ncluster " + str(i) + ": ")
        cent = kmodes[i,:]
        for j in equations_nom.columns[np.nonzero(cent)]:
            print(j)
'''




