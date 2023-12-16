from recommendations import sim_euclidean_distance, transformPrefs, calculateSimilarItems
from collections import defaultdict
import copy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from tfidf_with_film_descriptions import data

criticsOnlyFilms={'Armin': {'Wildlike': 6,'Shaun the Shep Movie':4,'Amy':1,'A Hard Day':6},
'Benjamin': {'Wildlike': 8,'Shaun the Shep Movie':4,'Amy':4,'A Hard Day':6},
'Caroline': {'Wildlike': 10,'Shaun the Shep Movie':4,'Amy':6,'A Hard Day':6},
'Doris': {'Wildlike': 8,'Shaun the Shep Movie':4,'Amy':3,'A Hard Day':4},
'Ernst': {'Wildlike': 9,'Shaun the Shep Movie':2,'Amy':4,'A Hard Day':5},
'Friedrich': {}
}

criticsOnlyConcerts={'Armin': {'Mumford&Sons': 6,'Bryan Adams':2,'Chris Cornell':4},
                     'Benjamin': {'Mumford&Sons': 6,'Bryan Adams':6,'Chris Cornell':8},
                     'Caroline': {'Mumford&Sons': 6,'Bryan Adams':2,'Chris Cornell':9},
                     'Doris': {'Mumford&Sons': 3,'Bryan Adams':3,'Chris Cornell':6},
                     'Ernst': {},
                     'Friedrich': {} }


def calc_similarities(ratings):
    similarities = defaultdict(lambda: {})
    for name1 in ratings:
        for name2 in ratings:
            if name1 == name2:
                continue
            similarities[name1][name2] = sim_euclidean_distance(ratings, name1, name2)
    return similarities

# FRIEDRICH'S RATING

movie_names = ['Wildlike', 'Shaun the Shep Movie', 'Amy', 'A Hard Day']
friedrich_ratings = [10, 10, 10, 10]
friedrich_dict = {}
for name in movie_names:
    for rating in friedrich_ratings:
        friedrich_dict[name] = rating

criticsOnlyFilms['Friedrich'] = friedrich_dict

# EUCLIDEAN DISTANCE

without_friedrich = copy.deepcopy(criticsOnlyFilms)
del without_friedrich['Friedrich']

# Based on the reviews of the four films:
similarities_without_friedrich = calc_similarities(without_friedrich)

# Which pair of users Armin, Benjamin, Caroline and Doris are the most similar?
max_value = 0
pair = ''
for sim, value in similarities_without_friedrich.items():
    most_similar = max(value, key=value.get)
    if value[most_similar] > max_value:
        max_value = value[most_similar]
        pair = sim + ' ' + most_similar
    print(sim, most_similar, value[most_similar])

print("Best overall: ", pair, max_value)

# Which are the most dissimilar concerning Armin, Benjamin, Caroline and Doris?

min_value = 1
pair = ''
for sim, value in similarities_without_friedrich.items():
    if sim == 'Friedrich':
        continue
    least_similar = min(value, key=value.get)
    if value[least_similar] < min_value:
        min_value = value[least_similar]
        pair = sim + ' ' + least_similar
    print(sim, least_similar, value[least_similar])

print("Worst overall: ", pair, min_value)

# Which is most similar to Ernst

print("most similar to Ernst: ", max(similarities_without_friedrich['Ernst'],
                                     key=similarities_without_friedrich['Ernst'].get))

# Which is most similar to Friedrich?

sim = calc_similarities(criticsOnlyFilms)
print("most similar to Friedrich: ", max(sim['Friedrich'], key=sim['Friedrich'].get))

# Using the similarity measure derived from the Euclidean distance introduced in the lectures.
# Which Concert should Ernst go to see, based on the film reviews?


def recommendConcert(name, concerts, similarity):
    max_concerts_rating = 0
    concert = ''
    for other in concerts:
        if other == name: continue
        s = similarity[name][other]

        for m, r in concerts[other].items():
            if r * s > max_concerts_rating:
                print(other, r, s, r*s)
                max_concerts_rating = r * s
                concert = m
    return concert

x = recommendConcert('Ernst', criticsOnlyConcerts, sim)
print("Ernst should visit", x)

# Which Concert should Friedrich go to see, based on the film reviews?

x = recommendConcert('Friedrich', criticsOnlyConcerts, sim)
print("Friedrich should visit", x)

# Heike goes to the cinema to see “A Hard Day”, but it is sold out. Based on the ratings which film should be
# recommended as most similar to “A Hard Day”?

similar = calculateSimilarItems(criticsOnlyFilms, n=1)
print("Heike should watch", similar['A Hard Day'])

# Calculate TF-IDF based on information you can find in the WWW

desc = pd.DataFrame(data)
desc = desc['overview']

cv = CountVectorizer(stop_words='english')
word_count_vector = cv.fit_transform(desc)

tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(word_count_vector)
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=sorted(cv.vocabulary_.keys()), columns=["idf_weights"])
print(df_idf.sort_values(by=['idf_weights']))

# Calculate similarities between all movies


# Recommend the most similar movie
# Repeat steps 2-4 with Pearson correlation coefficient.


