import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]

##################################################



##Step 1: Read CSV File
df = pd.read_csv("destination_dataset.csv")
# print(df)


##Step 2: Select Features
features = [ 'keywords' , 'climate ', 'genres', 'duration']


##Step 3: Create a column in DF which combines all selected features
for feature in features:
	df.fillna(" ", inplace = True)
def combine_features(row):
	try:
		return row['keywords'] + " " + row['climate'] + " " + row['genres'] + " " +row['duration'] 
	except:
		print ("Error:",row)
df["combined_features"]=df.apply(combine_features,axis=1)
# print ("combined features:",df["combined_features"].head())


##Step 4: Create count matrix from this new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])


##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim  = cosine_similarity(count_matrix)
destination_user_likes = "pokhara"

## Step 6: Get index of this movie from its title
destination_index = get_index_from_title(destination_user_likes)
similar_destinations = list(enumerate(cosine_sim[destination_index]))


## Step 7: Get a list of similar destinations in descending order of similarity score
sorted_similar_destinations= sorted(similar_destinations, key = lambda x : x[1],reverse=True)


## Step 8: Print titles of first 4 destination
i=0
for destination in sorted_similar_destinations:
	print(get_title_from_index(destination[0]))
	i=i+1
	if i>3:
		break