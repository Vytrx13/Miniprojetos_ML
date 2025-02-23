''' 
        Country: Name of the country.
        Density (P/Km2): Population density measured in persons per square kilometer.
        Abbreviation: Abbreviation or code representing the country.
        Agricultural Land (%): Percentage of land area used for agricultural purposes.
        Land Area (Km2): Total land area of the country in square kilometers.
        Armed Forces Size: Size of the armed forces in the country.
        Birth Rate: Number of births per 1,000 population per year.
        Calling Code: International calling code for the country.
        Capital/Major City: Name of the capital or major city.
        CO2 Emissions: Carbon dioxide emissions in tons.
        CPI: Consumer Price Index, a measure of inflation and purchasing power.
        CPI Change (%): Percentage change in the Consumer Price Index compared to the previous year.
        Currency_Code: Currency code used in the country.
        Fertility Rate: Average number of children born to a woman during her lifetime.
        Forested Area (%): Percentage of land area covered by forests.
        Gasoline_Price: Price of gasoline per liter in local currency.
        GDP: Gross Domestic Product, the total value of goods and services produced in the country.
        Gross Primary Education Enrollment (%): Gross enrollment ratio for primary education.
        Gross Tertiary Education Enrollment (%): Gross enrollment ratio for tertiary education.
        Infant Mortality: Number of deaths per 1,000 live births before reaching one year of age.
        Largest City: Name of the country's largest city.
        Life Expectancy: Average number of years a newborn is expected to live.
        Maternal Mortality Ratio: Number of maternal deaths per 100,000 live births.
        Minimum Wage: Minimum wage level in local currency.
        Official Language: Official language(s) spoken in the country.
        Out of Pocket Health Expenditure (%): Percentage of total health expenditure paid out-of-pocket by individuals.
        Physicians per Thousand: Number of physicians per thousand people.
        Population: Total population of the country.
        Population: Labor Force Participation (%): Percentage of the population that is part of the labor force.
        Tax Revenue (%): Tax revenue as a percentage of GDP.
        Total Tax Rate: Overall tax burden as a percentage of commercial profits.
        Unemployment Rate: Percentage of the labor force that is unemployed.
        Urban Population: Percentage of the population living in urban areas.
        Latitude: Latitude coordinate of the country's location.
        Longitude: Longitude coordinate of the country's location.

        https://www.kaggle.com/datasets/nelgiriyewithana/countries-of-the-world-2023


'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('world-data-2023.csv')


irrelevant_columns = ['Country', 'Abbreviation', 'Capital/Major City', 'Calling Code', 'Currency-Code', 'Official language', 'Largest city', 'Latitude', 'Longitude']
percent_columns = ['Agricultural Land( %)', 'Forested Area (%)', 'Gross primary education enrollment (%)', 'Gross tertiary education enrollment (%)', 'Out of Pocket Health Expenditure (%)', 'Population: Labor Force Participation (%)', 'Tax Revenue (%)',
                   'Total tax rate', 'Unemployment rate', 'CPI Change (%)']
dolar_columns = ['Minimum wage', 'Gasoline Price']
X = dataset.drop(columns=irrelevant_columns)

X = X.drop(columns=['GDP'])
# removing the '%' from the columns
for column in percent_columns:
    X[column] = X[column].str.replace('%', '').astype(float).astype(float)
    
for column in dolar_columns:
    X[column] = X[column].str.replace('$', '').str.replace(',', '').astype(float)
    
# removing commas from all x columns
for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = X[column].str.replace(',', '').astype(float)




y= dataset['GDP']
y = y.str.replace('$', '').str.replace(',', '').astype(float)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)
imputer2 = SimpleImputer(missing_values=np.nan, strategy='mean')
y = imputer2.fit_transform(y.values.reshape(-1, 1))


# print(X.head())
# print(y.head())
# #primeira linha de x
# print(X.iloc[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = XGBRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print("usando xgboost")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

scores = cross_val_score(regressor, X, y, cv=10, scoring='r2')
print(scores.mean())


#random forest
# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators=10, random_state=0)
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)
# print("usando random forest")
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
# print(r2_score(y_test, y_pred))


# # using ann

# # Normalizando os dados antes da divisão


# sc = StandardScaler()
# X_scaled = sc.fit_transform(X)

# # Dividindo os dados
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Achatar y para evitar problemas com o TensorFlow
# y_train = y_train.ravel()
# y_test = y_test.ravel()

# # Construindo a ANN
# ann = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(units=128, activation='relu'),
#     tf.keras.layers.Dense(units=128, activation='relu'),
#     tf.keras.layers.Dense(units=1)  # Saída para prever PIB
# ])

# # Compilando o modelo
# ann.compile(optimizer='adam', loss='mean_squared_error')

# # Treinando a ANN
# ann.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

# # Fazendo previsões
# y_pred = ann.predict(X_test).flatten()

# # Exibindo os resultados
# print("Usando ANN")
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
# print(f"R² Score: {r2_score(y_test, y_pred)}")
