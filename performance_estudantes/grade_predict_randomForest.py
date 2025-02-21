from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import time

# fonte : https://archive.ics.uci.edu/dataset/320/student+performance

def print_feature_importance(model, column_transformer):
    # Extrair os nomes das features após OneHotEncoding
    feature_names = column_transformer.get_feature_names_out()
    
    # Obter as importâncias das features do modelo
    feature_importance = model.feature_importances_

    # Criar um DataFrame com as features e suas importâncias
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Imprimir as features mais importantes
    print("\nFeature Importance:")
    print(importance_df)



def main():
    df1 = pd.read_csv("student-mat.csv", sep=";")
    df2 = pd.read_csv("student-por.csv", sep=";")
    student_df = pd.concat([df1, df2], axis=0)
    X = student_df.drop(["G3"], axis=1)
    y = student_df["G3"] # nota final

    # Colunas categóricas
    categorical_cols = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]
    ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), categorical_cols)], remainder="passthrough")
    X = ct.fit_transform(X)
    
    # Normalização
    sc = StandardScaler()
    X = sc.fit_transform(X)

    #obs: em geral feature scaling é feito após o split dos dados, mas para este caso específico, não há problema em fazer antes

    # Dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Ajuste de Hiperparâmetros via GridSearch
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=0), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Melhor modelo encontrado
    best_model = grid_search.best_estimator_
    
    # Predição
    y_pred = best_model.predict(X_test)

    # Avaliação
    print("Best Model R2 Score:", best_model.score(X_test, y_test))
    print("Best Model Mean Squared Error:", np.mean((y_pred - y_test)**2))

    # Importância das features
    print_feature_importance(best_model, ct)
    print("Best Model Parameters:", grid_search.best_params_)

if __name__ == "__main__":
    tempo_inicial = time.time()
    main()
    print("Tempo de execução:", time.time() - tempo_inicial)
