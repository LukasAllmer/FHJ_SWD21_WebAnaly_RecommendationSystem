#!/usr/bin/env python3

from math import sqrt
import tfidf_with_film_descriptions as tfidf
import recommendations as rec
from movies_and_critics import criticsOnlyFilms, criticsOnlyConcerts 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import pearsonr
import pandas as pd

criticsComplete = {}
for critic in criticsOnlyFilms.keys():
    criticsComplete[critic] = criticsOnlyFilms[critic].copy()
    criticsComplete[critic].update(criticsOnlyConcerts[critic])

def main():

    print("""
----------
    2
----------
          """)
    print("2.1 / 2.2 Which pair of users Armin, Benjamin, Caroline and Doris are the most similar / dissimilar?")
    onlyCriticsDict = criticsOnlyFilms.copy() 
    onlyCriticsDict.pop('Ernst')
    onlyCriticsDict.pop('Friedrich')
    for person in onlyCriticsDict.keys():
        print(f"Euclidean Distance for {person}:")
        rec.sort_by_similarity_euclidean(onlyCriticsDict ,person)
        print(f"Pearson similarity for {person}:")
        rec.sort_by_similarity_pearson(onlyCriticsDict ,person)

    print("\n--------------------------------")

    print("\n2.3 Which is most similar to Ernst?")
    print("Euclid")
    rec.sort_by_similarity_euclidean(criticsOnlyFilms,"Ernst")
    print("Pearson")
    rec.sort_by_similarity_pearson(criticsOnlyFilms,"Ernst")

    print("\n--------------------------------")

    print("\n2.4 Euclid - Which is most similar to Friedrich?")
    print("Euclid")
    rec.sort_by_similarity_euclidean(criticsOnlyFilms,"Friedrich")
    print("Pearson")
    rec.sort_by_similarity_pearson(criticsOnlyFilms,"Friedrich")

    print("""
----------
    3
----------
          """)
    print("3.1 Which Concert should Ernst go to see, based on the film reviews?")
    print("Euclid")
    print(rec.getRecommendations(criticsComplete, "Ernst", similarity=rec.sim_euclidean_distance))
    print("Pearson")
    print(rec.getRecommendations(criticsComplete, "Ernst", similarity=rec.sim_pearson_distance))

    print("\n--------------------------------")

    print("\n3.2 - Which Concert should Friedrich go to see, based on the film reviews?")
    print("Euclid")
    print(rec.getRecommendations(criticsComplete, "Friedrich", similarity=rec.sim_euclidean_distance))
    print("Pearson")
    print(rec.getRecommendations(criticsComplete, "Friedrich", similarity=rec.sim_pearson_distance))

    print("""
----------
    4
----------
          """)
    print("4. Which film should be recommended as most similar to “A Hard Day”?")
    print("\n4.1 Calculating TF-IDF scores for movies.")
    tfidf.calculate_tfidf_for_data(tfidf.data, 'title', 'overview')

    print("\n--------------------------------")

    print("\n4.2 Euclidean distance based on TF-IDF between all movies.")
    similarity_matrix, movie_names = tfidf.calculate_euclidean_dists(tfidf.data, 'title', 'overview')
    print(pd.DataFrame(similarity_matrix, index=movie_names, columns=movie_names))
    print("\n4.2 Pearson correlations based on TF-IDF between all movies.")
    tfidf.calculate_pearson_corr(tfidf.data, 'title', 'overview')
    print("\n4.2 Euclidean distance based on user ratings between all movies.")
    print(rec.calculateSimilarItems(criticsOnlyFilms))
    print("\n4.2 Pearson correlations based on user ratings between all movies.")
    print(rec.calculateSimilarItems(criticsOnlyFilms, 10, rec.sim_pearson_distance))

    print("\n--------------------------------")

    print("\n4.3 Recommendation based on TF-IDF Euclidean distance.")
    tfidf.calculate_most_similar_tfidf_movie_euclidean(tfidf.data, 'title', 'overview', 'A Hard Day')
    print("\n4.3 Recommendation based on TF-IDF Pearson similarity.")
    tfidf.calculate_most_similar_tfidf_movie_pearson(tfidf.data, 'title', 'overview', 'A Hard Day')
    print("\n4.3 Recommendation based on other users ratings (Euclidean).")
    movies = rec.calculateSimilarItems(criticsOnlyFilms)
    print(f"Most similar to \"A Hard Day\": {movies['A Hard Day'][0]}")

    print("\n4.3 Recommendation based on other users ratings (Pearson).")
    movies = rec.calculateSimilarItems(criticsOnlyFilms, 10, rec.sim_pearson_distance)
    print(f"Most similar to \"A Hard Day\": {movies['A Hard Day'][0]}")

    return 0
main()