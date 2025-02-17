import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer



def predict_titanic_survival(trainCSV, testCSV):
    train_df = pd.read_csv(trainCSV)
    test_df = pd.read_csv(testCSV)

    # primeiro treinar o modelo
    X_train = train_df.drop(["Survived", "PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    y_train = train_df["Survived"]


    # transformar as colunas categóricas em binárias
    categorical_cols = ["Sex", "Embarked"]
    ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), categorical_cols)], remainder="passthrough")
    X_train = ct.fit_transform(X_train)

    # feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(X_train, y_train)

    # **Importância das features**
    feature_names = ct.get_feature_names_out()
    feature_importance = classifier.feature_importances_

    # Criar um dataframe para melhor visualização
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:")
    print(importance_df)


    # predição
    X_test = test_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    X_test = ct.transform(X_test)
    X_test = sc.transform(X_test)
    y_pred = classifier.predict(X_test)

    # escrever um predict.csv com o passengerID e a predição
    passenger_id = test_df["PassengerId"]
    output = pd.DataFrame({"PassengerId": passenger_id, "Survived": y_pred})
    output.to_csv("predict.csv", index=False)
    print("Predictions saved to predict.csv")


def algumas_analises(trainCSV):
    # Carregar os dados
    train_df = pd.read_csv("train.csv")

    # Comparação de sobrevivência por sexo
    sns.countplot(data=train_df, x="Sex", hue="Survived")
    plt.title("Taxa de Sobrevivência por Sexo")
    plt.show()

    # Mostrar as taxas reais
    survival_rates = train_df.groupby("Sex")["Survived"].mean()
    print(survival_rates)

    sns.countplot(data=train_df, x="Pclass", hue="Survived")
    plt.title("Taxa de Sobrevivência por Classe")
    plt.show()

    sns.histplot(train_df, x="Age", hue="Survived", bins=20, kde=True)
    plt.title("Distribuição da Idade por Sobrevivência")
    plt.show()



if __name__ == "__main__":
    algumas_analises("train.csv")
    predict_titanic_survival("train.csv", "test.csv")


