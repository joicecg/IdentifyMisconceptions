import pandas as pd
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
from kneed import KneeLocator, DataGenerator
import math

print(f'Reading csv...')
prev = pd.read_csv(r'C:\Users\I502765\OneDrive - Associacao Antonio Vieira\Ciência da Computação\TCC\Practice\data\new-data-set\csv-finais\previousStep.csv', encoding='utf-8') 
wrong = pd.read_csv(r'C:\Users\I502765\OneDrive - Associacao Antonio Vieira\Ciência da Computação\TCC\Practice\data\new-data-set\csv-finais\wrongStep.csv', encoding='utf-8') 
print('...Complete')

equations = pd.concat([prev, wrong], axis = 1)

# Define relevant columns
col = []
for column in equations.columns:
    if column != 'previousStep' and column != 'wrongStep':
      col.append(column)

equations_nom = equations[col]

# Determine clusters
cost = []
K = range(2, 15)
for n in list(K):
    kmode = KModes(n_clusters=n, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(equations_nom)
    cost.append(kmode.cost_)


kn = KneeLocator(K, cost, curve='convex', direction='decreasing')
print(kn.elbow)

xint = range(2, math.ceil(15)+1)
plt.xticks(xint)
plt.xlabel('number of clusters k')
plt.ylabel('Cost')
plt.plot(K, cost, 'bx-')
plt.vlines(kn.elbow, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
  
'''
plt.plot(K, cost, 'bx-')
plt.xlabel('k clusters')
plt.ylabel('Cost')
'''
plt.title('Elbow Method For Optimal k')
plt.show()
