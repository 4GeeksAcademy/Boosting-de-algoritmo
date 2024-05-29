#from utils import db_connect
#engine = db_connect()

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Problema -> predecir en base a medidas diagnósticas si un paciente tiene o no diabetes.

# EDA
# Cargar conjunto de datos
total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv")

# Eliminar duplicados (no hay)
if total_data.duplicated().sum():
    total_data = total_data.drop_duplicates()
print(total_data.shape)
total_data.head()

# A través del análisis multivariante de variables numéricas he visto como las siguientes variables tienen números atípicos:
# BloodPressure = 0 y > 120 -> no es posible (en una persona viva), 120 es infato y no lo cuentas
# Glucose = 0 -> no posible (en una persona viva)
# BMI = 0 -> no posible
# SkinThickness -> 1 dato atípico
# Los elimino
total_data = total_data.drop(total_data[total_data["Glucose"] == 0].index)
total_data = total_data.drop(total_data[total_data["BloodPressure"] == 0].index)
total_data = total_data.drop(total_data[total_data["BloodPressure"] > 120].index)
total_data = total_data.drop(total_data[total_data["BMI"] == 0].index)
total_data = total_data.drop(total_data[total_data["SkinThickness"] > 80].index)

# Visualización relaciones de variables
plt.figure(figsize=(12, 6))

pd.plotting.parallel_coordinates(total_data, "Outcome", color = ("#E58139", "#39E581", "#8139E5"))

plt.show()

# Eliminación de nulos (no hay)
total_data.isnull().sum().sort_values(ascending=False)

# División del dataset
num_variables = ["Glucose", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"] # Variables escogidas para la predicción

X = total_data.drop("Outcome", axis = 1)[num_variables]
y = total_data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Entrenamiento del modelo (no se escalan las variables predictoras)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state = 42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Optimización
# Búsqueda de hiperparámetros por el método de malla
hyperparams = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(model, hyperparams, scoring = "accuracy", cv = 10)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

grid.fit(X_train, y_train)

print(f"Best hyperparameters: {grid.best_params_}")

# Aplicación de hiperparámetros y entrenamiento con los mismos
model = DecisionTreeClassifier(criterion = "entropy", max_depth = 5, min_samples_leaf = 4, min_samples_split = 2, random_state = 42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Cálculo de métricas
print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print("Precision: {}".format(precision_score(y_test, y_pred)))
print("Recall: {}".format(recall_score(y_test, y_pred)))
print("f1_score: {}".format(f1_score(y_test, y_pred)))

