import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.cluster import KMeans

#a = [[90,30,80],
#     [34,67,56],
#     [45,23,57]]
#
#print(np.array(a))
#
#k=2
#res = np.array(a)[:,k]
##print(str(res))

#plt.pie(df[columnname].value_counts(),label=["Yes","No"],autopct="%.2f")

df = pd.read_csv(r"C:\Downloads\diabetes.csv")
print(df)

scaler = MinMaxScaler(feature_range=(0,1))

rescaledx= scaler.fit_transform(df)

print(rescaledx)


scaler = StandardScaler()
rescaledx=scaler.fit_transform(df)

print(rescaledx)


print(df.isnull())


print(df.isnull().sum())


np.random.seed(42)
x = np.random.randint(100,size=(100,3))
df = pd.DataFrame(x)

print(df)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)

df_x_scaled = pd.DataFrame(x_scaled)

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_x_scaled)

explained_var_ratio = pca.explained_variance_
print(explained_var_ratio) 

cumulative_var_ratio = np.cumsum(explained_var_ratio)
print(cumulative_var_ratio)

plt.plot(range(1,len(cumulative_var_ratio)+1),cumulative_var_ratio,marker="o")
plt.show()




y_true = ['Cat'] * 10 + ['Dog'] * 12 + ['Horse'] * 10
print(y_true[:5])

# Predicted labels
y_pred = (
    ['Cat'] * 8 +
    ['Dog'] +
    ['Horse'] +
    ['Cat'] * 2 +
    ['Dog'] * 10 +
    ['Horse'] * 8 +
    ['Dog'] * 2
)
print(y_pred[:5])

classes = ["Cat","Dog","Horse"]

cm =confusion_matrix(y_true,y_pred,labels=classes)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes)


disp.plot(cmap=plt.cm.Greens)
plt.show()

print(classification_report(y_true,y_pred))


x = [5, 8, 12, 5, 4, 13, 15, 7, 10, 14]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x,y))

print(data)

kmeans=KMeans(n_clusters=2,n_init="auto")
kmeans.fit(data)

centroids=kmeans.cluster_centers_
labels=kmeans.labels_

plt.scatter(x,y,c=labels,cmap="viridis",label="Data Points")

plt.scatter(centroids)