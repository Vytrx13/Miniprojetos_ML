import numpy as np
import pandas as pd
import difflib
import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# fonte : https://www.kaggle.com/datasets/harshshinde8/movies-csv
def combine_features(row):
    return row["genres"] + " " + row["keywords"] + " " + row["cast"] + " " + row["director"] + " " + row["original_title"] + " " + row["tagline"]



def main():
    movies_df = pd.read_csv('movies.csv')
    relevant_features = ["genres", "keywords", "cast", "director", "original_title", "tagline"]
    movies_df = movies_df[relevant_features]

    # Preencher valores nulos
    for feature in relevant_features:
        movies_df[feature] = movies_df[feature].fillna('')

    # Função para combinar os valores de todas as colunas relevantes em uma única string


    movies_df["combined_features"] = movies_df.apply(combine_features, axis=1)

    # Criar um objeto de vetorizador de texto
    tfidf_vectorizer = TfidfVectorizer()  
    vectors = tfidf_vectorizer.fit_transform(movies_df["combined_features"])
    # print(vectors.shape)
    # print(vectors)

    similarity = cosine_similarity(vectors)
    
    filme_inicial = input('Escreva o nome de um filme que voce gosta : ')
    filme_inicial = filme_inicial.lower()

    # Encontrar o filme usando difflib
    filmes = movies_df["original_title"].tolist()
    filme_sugerido = difflib.get_close_matches(filme_inicial, filmes)
    # print(filme_sugerido)

    if not filme_sugerido:
        print("Nenhum filme encontrado. Tente novamente com outro nome.")
        return

    index = movies_df[movies_df["original_title"] == filme_sugerido[0]].index[0]
    similar_movies = list(enumerate(similarity[index]))

    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True) # Ordenar os filmes por ordem decrescente de similaridade
    #lambda x: x[1] é uma função que retorna o segundo elemento de x
    #sorted_similar_movies é uma lista de tuplas (index, similarity)

    print("Filmes sugeridos para você:")
    i = 0

    for filme in sorted_similar_movies:
        j = filme[0]
        filme = movies_df[movies_df.index == j]["original_title"].values[0]
        if i < 10:
            print(f"{i+1}. {filme}")
            i += 1



    













if __name__ == "__main__":
    main()