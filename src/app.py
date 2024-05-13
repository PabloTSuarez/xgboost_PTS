#from utils import db_connect
#engine = db_connect()
# your code here
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv')
df

X = df.drop(columns=['Outcome'],axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

accuracy = accuracy_score(y_test, y_pred)
accuracy

#Grilla de hiperparametros
param_grid = {
    'n_estimators': [100,200,300],
    'max_depth' : [3,4,5],
    'learning_rate' : [0.1,0.01,0.05,0.001]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)

#Entrenar grilla
grid_search.fit(X_train,y_train)

#obtener los mejores hiperparametros
grid_search.best_params_

#Obtener el mejor modelo
best_model = grid_search.best_estimator_

#Hacer predicciones con el mejor modelo
y_pred = best_model.predict(X_test)

#Calcular la precision del mejor modelo
accuracy = accuracy_score(y_test,y_pred)
accuracy

#Grilla con distintos modelos e hiperparametros
param_grid = {
    'n_estimators': [100,200,300],
    'max_depth' : [3,4,5],
    'learning_rate' : [0.1,0.01,0.05,0.001],
    'subsample' : [0.5,0.7,1],
    'colsample_bytree' : [0.5,0.7,1]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)

#Entrenar la grilla
grid_search.fit(X_train,y_train)

#Obtener los mejores hiperparametros
grid_search.best_params_

#Obtener el mejor modelo
best_model = grid_search.best_estimator_

#Hacer predicciones con el mejor modelo
y_pred = best_model.predict(X_test)

#Calcular la precision del mejor modelo
accuracy = accuracy_score(y_test,y_pred)
accuracy

model.save_model('model.json')