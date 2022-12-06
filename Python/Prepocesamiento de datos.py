# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:24:06 2022

@author: serra
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

detalle_credito=pd.read_csv("credit_record.csv")
detalle_cliente = pd.read_csv('application_record.csv')

#Se actualiza columna "days birth" por la edad:
detalle_cliente['DAYS_BIRTH']=round(detalle_cliente['DAYS_BIRTH']/-365)

#Se actualiza comlumna 'days_employed' por tiempo de trabajo
#Se necesita que almenos el empleado tenga 6 meses en el trabado para que se le considere

detalle_cliente['DAYS_EMPLOYED']=detalle_cliente['DAYS_EMPLOYED'].replace([365243],-1)
detalle_cliente['DAYS_EMPLOYED']=round(detalle_cliente['DAYS_EMPLOYED']/-365)

col_esp=['ID', 'GENERO', 'TIENE_AUTO', 'TIENE_PROPIEDAD',
         'HIJOS', 'INGRESOS_ANUALES', 'CATEGORIA_DE_INGRESOS',
          'NIVEL_EDUCATIVO', 'ESTADO_CIVIL', 'ESTADO_DE_PROPIEDAD',
          'EDAD', 'TIEMPO_EMPLEO', 'TEL_CEL', 'TEL_TRABAJO',
          'TEL_FIJO', 'CORREO', 'OCUPACION', 'DEPENDIENTES ECONOMICOS']

detalle_cliente.columns=col_esp

col=list(detalle_cliente.columns.values)
no_deseados=['ID','INGRESOS_ANUALES','HIJOS','TEL_CEL', 'TEL_TRABAJO','TEL_FIJO','CORREO','DEPENDIENTES ECONOMICOS','EDAD','TIEMPO_EMPLEO']
for rmve in no_deseados:
  col.remove(rmve)
col

print(detalle_cliente['GENERO'].value_counts())
detalle_cliente['GENERO']=detalle_cliente['GENERO'].replace(['F'],1)
detalle_cliente['GENERO']=detalle_cliente['GENERO'].replace(['M'],0)
#MUJER = 1
#HOMBRE = 0

detalle_cliente['TIENE_AUTO']=detalle_cliente['TIENE_AUTO'].replace(['Y'],1)
detalle_cliente['TIENE_AUTO']=detalle_cliente['TIENE_AUTO'].replace(['N'],0)

detalle_cliente['TIENE_PROPIEDAD']=detalle_cliente['TIENE_PROPIEDAD'].replace(['Y'],1)
detalle_cliente['TIENE_PROPIEDAD']=detalle_cliente['TIENE_PROPIEDAD'].replace(['N'],0)

detalle_cliente['CATEGORIA_DE_INGRESOS']=detalle_cliente['CATEGORIA_DE_INGRESOS'].replace(['Working'],1)
detalle_cliente['CATEGORIA_DE_INGRESOS']=detalle_cliente['CATEGORIA_DE_INGRESOS'].replace(['Commercial associate'],2)
detalle_cliente['CATEGORIA_DE_INGRESOS']=detalle_cliente['CATEGORIA_DE_INGRESOS'].replace(['Pensioner'],3)
detalle_cliente['CATEGORIA_DE_INGRESOS']=detalle_cliente['CATEGORIA_DE_INGRESOS'].replace(['State servant'],4)
detalle_cliente['CATEGORIA_DE_INGRESOS']=detalle_cliente['CATEGORIA_DE_INGRESOS'].replace(['Student'],5)

detalle_cliente['NIVEL_EDUCATIVO']=detalle_cliente['NIVEL_EDUCATIVO'].replace(['Secondary / secondary special'],1)
detalle_cliente['NIVEL_EDUCATIVO']=detalle_cliente['NIVEL_EDUCATIVO'].replace(['Higher education'],2)
detalle_cliente['NIVEL_EDUCATIVO']=detalle_cliente['NIVEL_EDUCATIVO'].replace(['Incomplete higher'],3)
detalle_cliente['NIVEL_EDUCATIVO']=detalle_cliente['NIVEL_EDUCATIVO'].replace(['Lower secondary'],4)
detalle_cliente['NIVEL_EDUCATIVO']=detalle_cliente['NIVEL_EDUCATIVO'].replace(['Academic degree'],5)


detalle_cliente[col[5]]=detalle_cliente[col[5]].replace(['Married'],1)
detalle_cliente[col[5]]=detalle_cliente[col[5]].replace(['Single / not married'],2)
detalle_cliente[col[5]]=detalle_cliente[col[5]].replace(['Civil marriage'],3)
detalle_cliente[col[5]]=detalle_cliente[col[5]].replace(['Separated'],4)
detalle_cliente[col[5]]=detalle_cliente[col[5]].replace(['Widow'],5)

detalle_cliente[col[6]]=detalle_cliente[col[6]].replace(['House / apartment'],1)
detalle_cliente[col[6]]=detalle_cliente[col[6]].replace(['With parents'],2)
detalle_cliente[col[6]]=detalle_cliente[col[6]].replace(['Municipal apartment'],3)
detalle_cliente[col[6]]=detalle_cliente[col[6]].replace(['Rented apartment'],4)
detalle_cliente[col[6]]=detalle_cliente[col[6]].replace(['Office apartment'],5)
detalle_cliente[col[6]]=detalle_cliente[col[6]].replace(['Co-op apartment'],6)



detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['Laborers'],1)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['Core staff'],2)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['Sales staff'],3)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['Managers'],4)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['Drivers'],5)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['High skill tech staff'],6)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['Accountants'],7)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['Medicine staff'],8)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['Cooking staff'],9)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['Security staff'],10)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['Cleaning staff'],11)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['Private service staff'],12)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['Low-skill Laborers'],13)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['Secretaries'],14)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['Waiters/barmen staff'],15)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['Realty agents'],16)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['HR staff'],17)
detalle_cliente[col[7]]=detalle_cliente[col[7]].replace(['IT staff'],18)

# =============================================================================
# Para no tener valores outliers, trataremos a los valores de status X:No hay prestamos 
# y C:Pagado este mes, lo cambiaremos a 0: 0-29 dias de atraso
# =============================================================================

detalle_credito['STATUS']=detalle_credito['STATUS'].replace(['X'],0)
detalle_credito['STATUS']=detalle_credito['STATUS'].replace(['C'],0)

#Se convierte la columna 'STATUS' en entero
detalle_credito['STATUS'] = detalle_credito['STATUS'].apply(pd.to_numeric)

# LA BASE DE DATOS detalle_credito tenemos datos repetidos del clientes ya que es a lo largo
# de historia de años del cliente, entonces se tomara del ultimo mes y ademas no se cuenta con
# la informacion de todos los clientes

detalle_credito = detalle_credito.groupby('ID')['STATUS'].max().reset_index()

detalle_credito.groupby('ID')['STATUS'].count().reset_index()


# Se une la base de datos detalle_cliente y detalle_credito con el status 
# de los clientes que se cuenta (como observacion la base datos disminuyo bastante)
clientes = pd.merge(detalle_cliente, detalle_credito, left_on='ID', right_on='ID')
#Se agrega la columna 'ALTO_RIESGO'
#EL STATUS ESTA DADO POR
"""
0: 1-29 days past due
1: 30-59 days past due
2: 60-89 days overdue 
3: 90-119 days overdue 
4: 120-149 days overdue 
"""
#SUPONDREMOS QUE LA EMPRESA DECIDE TOMAR A LOS CLIENTES COMO MOROSO CUANDO EL CLIENTE NO HA PAGADO COMO MAXIMO 3 MESES
clientes['MOROSO']=np.where(clientes['STATUS']<3,0,1)
clientes

clientes.to_csv('clientes.csv',index=False)