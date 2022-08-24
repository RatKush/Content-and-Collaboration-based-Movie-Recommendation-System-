import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append(r"C:\mrep_code_heroku")

###### helper functions. Use them when needed #################
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]

# details one by one
def get_detail(index,detail):
	return df[df.index==index][detail].values[0]

#details in once
def get_details(index):
	details= []
	details.append(df[df.index==index]["vote_average"].values[0])
	details.append(df[df.index==index]["genres"].values[0])
	details.append(df[df.index==index]["cast"].values[0])
	details.append(df[df.index==index]["director"].values[0])
	return details
#############################################################

##Step 1: Read CSV File
#movie_datase= "C:\mrep_code_heroku\content_base\movie_dataset.csv"
df= pd.read_csv("movie_dataset.csv")
#print (df.head())
#print (df.columns)

##Step 2: Select Features
features= ['keywords', 'cast', 'genres', 'director']

##Step 3: Create a column in DF which combines all selected features 
#fn  takes a rowhead and return given its features
for feature in features:
	df[feature]= df[feature].fillna('')

def combined_features(row):
	try:
		return row['keywords']+" "+row['cast']+" "+row["genres"]+" "+row["director"]
	except:
		#print ("error",row)
		print ("ERROR")

df["combined_features"]= df.apply(combined_features, axis=1)
#print ("combined_features: ", df["combined_features"].head())

##Step 4: Create count matrix from this new combined column
cv= CountVectorizer()
count_matrix= cv.fit_transform(df["combined_features"])

##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim= cosine_similarity(count_matrix)

##############################################################################################################################
movie_user_likes = "Interstellar"
print ("You liked:",movie_user_likes)
##############################################################################################################################

## Step 6: Get index of this movie from its title
# first get that row
movie_index= get_index_from_title(movie_user_likes)
#print (get_detail(movie_index,"cast"))   this is right sintex
#print ("rating:",get_detail(movie_index,"vote_average"))
#print ("genres:",get_detail(movie_index,"genres"))
#print ("cast:",get_detail(movie_index,"cast"))
#print ("director:",get_detail(movie_index,"director"))
movie_details= get_details(movie_index)
print ("rating:",movie_details[0])
print ("genres:",movie_details[1])
print ("cast:",movie_details[2])
print ("director:",movie_details[3])
print ("similar movies")
#go inside cosine_sim &  enumerate that row adds iteration count
similar_movies=  list(enumerate(cosine_sim[movie_index]))

## Step 7: Get a list of similar movies in descending order of similarity score
# sort key is second ele of similar_movies i.e scores
sorted_similar_movies= sorted(similar_movies, key= lambda x:x[1], reverse= True)

## Step 8: Print titles of first 50 movies
i=0
for ele in sorted_similar_movies:
	print (get_title_from_index(ele[0]))
	i=i+1
	if(i>50):
		break
