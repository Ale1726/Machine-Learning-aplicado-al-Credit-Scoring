# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:31:04 2022

@author: serra
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn import tree
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,precision_score,recall_score
#from string import ascil_uppercase
import pandas as pd
import seaborn as sns

clientes = pd.read_csv('clientes.csv')

columnas= list(clientes.columns.values)
no_deseados=['MOROSO','ESTADO_CIVIL','TEL_CEL','TEL_TRABAJO','TEL_FIJO',
             'CORREO','STATUS','ID']
for no_deseado in no_deseados:
    columnas.remove(no_deseado)


clientes_entrena = clientes[['EDAD','INGRESOS_ANUALES','NIVEL_EDUCATIVO','ESTADO_CIVIL','DEPENDIENTES ECONOMICOS']]

clientes_prueba = clientes['MOROSO']

datos_entrena, datos_prueba, clase_entrena, clase_prueba = train_test_split(
    clientes_entrena,clientes_prueba,test_size=0.30,random_state = 42)

arbol_decision= tree.DecisionTreeClassifier(
    criterion='entropy', #Criterio
    max_depth=5) #Mxima profundidad

arbol = arbol_decision.fit(datos_entrena,clase_entrena)

leyendas=list(datos_entrena.columns.values)
print(tree.export_text(arbol,feature_names=leyendas)) 

plt.figure(figsize=(60,40))

tree.plot_tree(arbol,feature_names=leyendas,fontsize=15)

plt.show()

y_predict = arbol.predict(datos_prueba)
acuaracy=arbol_decision.score(datos_prueba,clase_prueba)
exactitud=accuracy_score(clase_prueba,y_predict)
sensibilidad=recall_score(clase_prueba,y_predict)
print(acuaracy)
print(exactitud)
print(sensibilidad)

reporte=classification_report(clase_prueba, y_predict)
print(reporte)