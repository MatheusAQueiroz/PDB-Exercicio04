# Imports principais
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import sys
# Imports sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import dataset
dataset = pd.read_csv(r'weatherAUS.csv')

# Selecionando cinco colunas e a variável dependente e apagando as linhas com dados faltantes
dataset = dataset[['Location','MinTemp','MaxTemp','Rainfall','WindGustSpeed','RainTomorrow']]
dataset = dataset.dropna()

# Valores dependentes e independentes
independent = dataset.iloc[:, :-1].values
dependent = dataset.iloc[:, -1].values

# Scalers para normalização de dados
standardScaler = StandardScaler()
independent[:, 1:5] = standardScaler.fit_transform(independent[:, 1:5])

# One Hot Encoder para o vento e local
transformer = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
independent = transformer.fit_transform(independent)

# Label encoder para os dados dependentes
labelEncoder = LabelEncoder()
dependent = labelEncoder.fit_transform(dependent)

# Train Test Split com fator 80% - 20%
ind_train, ind_test, dep_train, dep_test = train_test_split(independent, dependent, test_size = 0.2, random_state = 1)

# Predição de dados
linearRegression = LinearRegression()
linearRegression.fit(ind_train, dep_train)
pred_train = linearRegression.predict(ind_test)

# Print dos dados
np.set_printoptions(precision=2)
expected = dep_test.reshape(len(dep_test), 1)
predict  = pred_train.reshape(len(pred_train), 1)
print(np.concatenate((expected, predict), axis=1))