import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

metadata = pd.read_csv('datafiles/movies_metadata.csv', low_memory=False)
# Remove some rows with bad IDs
metadata = metadata.drop([19730, 29503, 35587])
credits = pd.read_csv('datafiles/credits.csv')
keywords = pd.read_csv('datafiles/keywords.csv')

# Convert IDs to int
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# Merge keyword and credits into main metadata dataframe

metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

# Parse the stringified features into their corresponding python objects
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)


# we need to get the director's name from crew feature
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# return the top 3 elements or the entire list
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


# find director in metadata['crew']
metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

print(metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3))


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

metadata['overview'] = metadata['overview'].fillna('')


# join all of the features into on soup(become a new 'overview')
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres']) + ' ' + x['overview']


metadata['soup'] = metadata.apply(create_soup, axis=1)

print(metadata[['soup']].head(2))

# Now ,all we have to do is to get similarity of the soup
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# get the index from title
metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])


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
