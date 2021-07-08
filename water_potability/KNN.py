import pandas as pd
import numpy as np
from  sklearn.neighbors  import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
way_to_file = "water_potability.csv"
my_data = pd.read_csv(way_to_file)
my_data = my_data.dropna()
#my_data = my_data.drop(["index"],axis = 1)
my_data = my_data.reset_index()
my_data = my_data.drop(["index"],axis = 1)
y = my_data["Potability"]
x = my_data.drop(["Potability"], axis = 1)
X_train,X_test,Y_train,Y_test = train_test_split(x,y,train_size = 0.8,random_state = 3)
features = X_train.columns.tolist()
accuracy = 0
current_features = []
for i in features:
    current_features.append(i)
    X2_train = X_train[current_features]
    X2_test = X_test[current_features] 
    KNN_model =  KNeighborsClassifier(n_neighbors = 5)
    KNN_model.fit(X2_train,Y_train)
    Y_pred = KNN_model.predict(X2_test)
    print(accuracy_score(Y_pred,Y_test))
    new_accuracy = accuracy_score(Y_pred,Y_test)
    if(new_accuracy > accuracy):
        accuracy = new_accuracy
    else:
        current_features.pop()

print(current_features)
X2_train = X_train[current_features]
X2_test = X_test[current_features]
max_index = 5
i = 1
while i < 20:
    KNN_model = KNeighborsClassifier(n_neighbors = i)
    KNN_model.fit(X2_train,Y_train)
    Y_pred = KNN_model.predict(X2_test)
    new_accuracy = accuracy_score(Y_pred,Y_test)
    print(new_accuracy)
    if(new_accuracy >accuracy):
        max_index = i 
    i += 1
