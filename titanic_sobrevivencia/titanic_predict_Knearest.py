import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def predict_titanic_survival(trainCSV, testCSV):
    train_df = pd.read_csv(trainCSV)
    test_df = pd.read_csv(testCSV)

    # Remover colunas irrelevantes
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    X_train = train_df.drop(columns=["Survived"] + drop_cols)
    y_train = train_df["Survived"]
    X_test = test_df.drop(columns=drop_cols)


   
    imputer_mean = SimpleImputer(strategy="mean")
    imputer_most_frequent = SimpleImputer(strategy="most_frequent")

    numeric_cols = ["Age", "Fare"]
    categorical_cols = ["Sex", "Embarked"]

    # Preencher valores faltantes nas colunas numéricas
    X_train[numeric_cols] = imputer_mean.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = imputer_mean.transform(X_test[numeric_cols])

    # Preencher valores faltantes nas colunas categóricas
    X_train[categorical_cols] = imputer_most_frequent.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = imputer_most_frequent.transform(X_test[categorical_cols])

    # Transformar colunas categóricas em binárias
    ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), categorical_cols)], remainder="passthrough")
    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)  # Aqui não usamos fit_transform, apenas transform

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Treinar o modelo
    classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    classifier.fit(X_train, y_train)

    # Predição
    y_pred = classifier.predict(X_test)

    # Gerar arquivo CSV de saída
    passenger_id = test_df["PassengerId"]
    output = pd.DataFrame({"PassengerId": passenger_id, "Survived": y_pred})
    output.to_csv("predict_Knearest.csv", index=False)
    print("Predictions saved to predict_Knearest.csv")


if __name__ == "__main__":
    predict_titanic_survival("train.csv", "test.csv")
