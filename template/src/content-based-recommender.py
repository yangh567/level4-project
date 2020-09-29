# Content-based Recommender
# build a system that recommends movies that are similar to a particular movie.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


metadata = pd.read_csv('datafiles/movies_metadata.csv', low_memory=False)
print(metadata['overview'].head())

# you need to extract some kind of features from
# the above text data before you can compute the
# similarity and/or dissimilarity between them

# compute Term Frequency-Inverse Document Frequency (TF-IDF score)for each document
# TF-IDF score(word occurring in document)


# create the TF-IDF Vectorizer object,remove all english stop words(the.a)
tfidf = TfidfVectorizer(stop_words='english')

# replace the Na with empty string
metadata['overview'] = metadata['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# 75,827 different vocabularies or words in your dataset have 45,000 movies.
print(tfidf_matrix.shape)
# print(tfidf.get_feature_names())

# calculate the cosine similarity matrix
# 45466x45466 movies matrix

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# print(cosine_sim.shape)

# Construct reverse map of indices and movie titles
# get the title's indexes
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity of all movies with this movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies based on similarity scores with descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # get the most 10 similar movies
    sim_scores = sim_scores[1:11]

    # get the 10 of similar movies's indices
    movie_indices = [i[0] for i in sim_scores]

    # return 10 most similar movies
    return metadata['title'].iloc[movie_indices]


print(get_recommendations('The Dark Knight Rises'))
