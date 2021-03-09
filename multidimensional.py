import pandas as pd 
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA 
import seaborn as sns 
from matplotlib import pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d.axes3d import Axes3D

equation_data = pd.read_csv("../data/multidimensional.csv", encoding = 'utf-8', 
							index_col = ["previousStep"]) 

# print first 5 rows
print(equation_data.head())

clusters = 2

kmeans = KMeans(n_clusters = clusters) 
kmeans.fit(equation_data) 

print(kmeans.labels_)

# generating correlation heatmap 
sns.heatmap(equation_data.corr(), annot = True) 

# posting correlation heatmap to output console 
plt.show() 

# generating correlation data 
df = equation_data.corr() 
df.index = range(0, len(df)) 
df.rename(columns = dict(zip(df.columns, df.index)), inplace = True) 
df = df.astype(object) 

''' Generating coordinates with 
corresponding correlation values '''
for i in range(0, len(df)): 
	for j in range(0, len(df)): 
		if i != j: 
			df.iloc[i, j] = (i, j, df.iloc[i, j]) 
		else : 
			df.iloc[i, j] = (i, j, 0) 

df_list = [] 

# flattening dataframe values 
for sub_list in df.values: 
	df_list.extend(sub_list) 

# converting list of tuples into trivariate dataframe 
plot_df = pd.DataFrame(df_list) 

fig = plt.figure() 
ax = Axes3D(fig) 

# plotting 3D trisurface plot 
ax.plot_trisurf(plot_df[0], plot_df[1], plot_df[2], 
					cmap = cm.jet, linewidth = 0.2) 

plt.show() 



