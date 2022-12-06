# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:32:58 2022

@author: serra
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn import tree
from random import sample
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
#from string import ascil_uppercase
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

clientes = pd.read_csv('clientes.csv')


columnas= list(clientes.columns.values)
no_deseados=['MOROSO','TEL_CEL','TEL_TRABAJO','TEL_FIJO','TIENE_AUTO',
             'CORREO','STATUS','ID','OCUPACION']
for no_deseado in no_deseados:
    columnas.remove(no_deseado)


#Se eligue 2/3 de los datos para crear arboles de decisiones aleatorios
clientes.sample(frac=2/3,replace=True)

print(clientes.columns.values)
print(columnas)
CLIENTES=clientes[columnas]
sample(set(CLIENTES.columns[:-1]),5)


bosque = RandomForestClassifier(n_estimators=100,
                                criterion='entropy',#entropy
                                max_features= 'sqrt',
                                bootstrap=True,
                                max_samples=2/3, #muestro de los datos
                                oob_score=True  #evaluamos con las instancias que no fueron ocupadas
                                )



a=CLIENTES[CLIENTES.columns[:-1].values]
b=clientes['MOROSO']
#75 de entrenamiento, 25 prueba
x_entrena,x_prueba,y_entrena,y_prueba=train_test_split(a,b,random_state=42)
x_entrena
bosque.fit(x_entrena,y_entrena)

cliente_aleatorio=x_entrena.sample()
print(cliente_aleatorio)

x_prueba
y_predict=bosque.predict(x_prueba)
acuaracy_1=accuracy_score(y_prueba,y_predict)
print(acuaracy_1)

reporte=classification_report(y_prueba, y_predict)
print(reporte)