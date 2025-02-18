import pandas as pnd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler  
from sklearn.cluster import KMeans

try:
    training_data = pnd.read_csv("Mall_Customers.csv")
except FileNotFoundError:
    print("No such file exist please put the Mall_Customers.csv File in the folder with main.py")

allspend = training_data[["Age","Annual Income (k$)","Spending Score (1-100)"]]

sl = StandardScaler()
sd = sl.fit_transform(allspend)

ssel = []
for k in range(1,20):
    km = KMeans(n_clusters=k)
    km.fit(sd)
    ssel.append(km.inertia_)

cluster = KMeans(n_clusters=5)
tl = cluster.fit_predict(sd)
training_data['clusters'] = tl
fig = plt.figure()
ax = plt.axes(projection='3d')

for i in range(5):
    cluster_data = training_data[training_data['clusters'] == i]
    ax.scatter(cluster_data['Age'], cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], label=f'Cluster {i}')

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title('K-Means Clustering (Age, Income, Spending Score)')
ax.legend()
plt.show()
