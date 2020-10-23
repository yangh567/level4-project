import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# 1.read data ,from source
course_df = pd.read_csv("datafiles/courses.csv")

# 2. drop rows with NaN values for 'Description' column,
# cause no description wont be of much use
course_df = course_df.dropna(how='any')

# 3.pre-processing step: remove words like we'll you'll,they'll
course_df['Description'] = course_df['Description'].replace({"'ll": " "}, regex=True)
# 4.another pre-processing step:Removal of '-' from CourseId field
course_df['CourseId'] = course_df['CourseId'].replace({"-": " "}, regex=True)

# 5.combine 3 columns namely:CourseId.CourseTitle,Description
comb_frame = course_df.CourseId.str.cat(" " + course_df.CourseTitle.str.cat(" " + course_df.Description))
# 6.remove all characters except number and alphabets
comb_frame = comb_frame.replace({"[^A-Za-z0-9]+": ""}, regex=True)

# print(comb_frame.head(1))
# 6.convert our textual data into vector matrix
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(comb_frame)

# true_k,derived from elbow method
true_k = 30

# running model with 15 different centroid
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=500, n_init=15)
model.fit(X)

# Top term in each clusters
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()

# print(order_centroids[1,:15])
# print(terms[order_centroids[1,:15][0]])
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :15]:
        print(' %s' % terms[ind]),
    print

# clusters derived from this approach still can be
# improved by further division into other clusters to
# derive out these smaller course categories with less number of
# courses. For, these further divisions
# which can be formulated as optimization problem
# with error minimization.
# We don’t want to over-fit our model because of which,
# we’ll use ‘elbow-test’ method for finding ideal value of k.
# The idea is whenever a sharp drop in error comes for a given value of ‘k’,
# that value is good enough for forming clusters. These formed clusters
# will have sharp minima in error
# 利用error minimization optimization 来找到 idea value of k 来分clusters

# this is the data structure to store Sum-Of-Square-Errors
sse = {}

# Looping over multiple values of k from 1 to 40
for k in range(1, 40):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100).fit(X)
    comb_frame["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_

# Plotting the curve with 'k'-value vs SSE
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
#plt.savefig('elbow_method.png')

# Save machine learning model
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
