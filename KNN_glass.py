#import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
#load data
df = pd.read_csv('C:/Users/HP/PycharmProjects/Excelrdatascience/glass.csv')
df.head()
df.tail()
# value count for glass types
df.Type.value_counts()
#Data exploration and visualizaion
# correlation matrix
cor = df.corr()
sns.heatmap(cor)
#pairwise plot of all the features
sns.pairplot(df,hue='Type')
plt.show()
scaler = StandardScaler()
scaler.fit(df.drop('Type',axis=1))
StandardScaler()
#perform transformation
scaled_features = scaler.transform(df.drop('Type',axis=1))
scaled_features
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()
#Declare feature vector and target variable¶
X = df_feat.values
y = df['Type'].values
X.shape
y.shape
#Split data into separate training and test set
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.3,random_state=45)
#setting random state ensures split is same eveytime, so that the results are comparable
#Model training
knn = KNeighborsClassifier(n_neighbors=4,metric='manhattan')
classifer = KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2 )
classifer.fit(X_train,y_train)
KNeighborsClassifier()
y_pred= classifer.predict(X_test)
y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
ax = sns.heatmap(cm, annot=True, fmt="d")
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
k_range = range(1, 25)
k_scores = []
error_rate = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # kscores - accuracy
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

    # error rate
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error_rate.append(np.mean(y_pred != y_test))

# plot k vs accuracy
plt.plot(k_range, k_scores)
plt.xlabel('value of k - knn algorithm')
plt.ylabel('Cross validated accuracy score')
plt.show()

# plot k vs error rate
plt.plot(k_range, error_rate)
plt.xlabel('value of k - knn algorithm')
plt.ylabel('Error rate')
plt.show()
