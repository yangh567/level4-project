import pandas as pd

metadata = pd.read_csv('datafiles/movies_metadata.csv', low_memory=False)
print(metadata.columns)

# c is the mean vote across whole project
c = metadata['vote_average'].mean()
print(c)

# Number of the votes,m,received by movie in 90th percentile
m = metadata['vote_count'].quantile(0.90)
print(m)

# now we filter out the movies that has the votes < 160,we use .copy() to ensure new data wont affect original data
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
print('This is all of the movies:', metadata.shape)
print('This is number of the movies have the votes >= 160', q_movies.shape)


# number of the votes of those movie that satisfied the requirement >= 160

def weighted_rating(m1, c1, x):
    v = x['vote_count']
    r = x['vote_average']

    return (v / (v + m1)) * r + (m1 / (m1 + v)) * c1


# adding the score to the dataframe
q_movies['score'] = weighted_rating(m, c, q_movies)
# sort the score with descending order
q_movies = q_movies.sort_values('score',ascending=False)
print(q_movies[['title','vote_count','vote_average','score']].head(20))
