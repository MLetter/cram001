# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:41:12 2023

@author: mstte
"""

import pandas as pd
import numpy as np



df=pd.read_csv("train2.csv",low_memory=False)
dfo=df.copy()


print(df)

print(df.info)

print(df.isnull().sum())

df[df['Age'].str.contains('_')]


#fix age. remove _ and rows with ages over 99
df['Age']=df['Age'].str.replace('_', '')

df = df[df['Age'].str.len()==2]



#fix income. remove _ 
df['Annual_Income']=df['Annual_Income'].str.replace('_', '')


#fix age. remove _ and rows with ages over 99
df['Num_of_Loan']=df['Num_of_Loan'].str.replace('_', '')

df = df[df['Num_of_Loan'].str.len()==1]
print(df['Num_of_Loan'])

#fix income. remove _ 
df['Outstanding_Debt']=df['Outstanding_Debt'].str.replace('_', '')

df['Amount_invested_monthly']=df['Amount_invested_monthly'].str.replace('_', '')


df.to_csv('file_name.csv')