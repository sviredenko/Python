KNimport pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

data = pd.read_csv('Iris.csv')
data.drop(["Id"],axis = 1,inplace = True)
x = data.iloc[:,:-1]
y = data[ "Species"]
X_train, X_test,Y_train, Y_test = train_test_split(x.to_numpy(),y.to_numpy(),test_size = 0.2,random_state = 51)
KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train,Y_train)
Y_pred = KNN_model.predict(X_test)
print(accuracy_score(Y_pred,Y_test))
