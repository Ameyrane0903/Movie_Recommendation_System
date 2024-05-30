import numpy as np
import pandas as pd
import ast
import nltk
import pickle

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits , on = 'title')

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

movies.dropna(inplace = True) #Removes Null Values

def convert(obj):
  L=[]
  for i  in ast.literal_eval(obj):
    L.append(i['name'])
  return L

movies['genres'] = movies['genres'].apply(convert)

movies['keywords'] = movies['keywords'].apply(convert)

def convert3(obj):
  L=[]
  count = 0
  for i  in ast.literal_eval(obj):
    if(count != 3):
      L.append(i['name'])
      count += 1
    else:
      break
  return L

movies['cast'].apply(convert3)

movies['cast'] = movies['cast'].apply(convert3)

def convert_crew(obj):
  L=[]
  for i  in ast.literal_eval(obj):
    if(i['job'] == 'Director'):
      L.append(i['name'])
      break
  return L

movies['crew'] = movies['crew'].apply(convert_crew)

movies.rename(columns={'crew': 'Director'}, inplace=True)

movies['overview'] = movies['overview'].apply(lambda x : x.split())

movies['genres'] = movies['genres'].apply(lambda x : [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x : [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x : [i.replace(" ", "") for i in x])
movies['Director'] = movies['Director'].apply(lambda x : [i.replace(" ", "") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['Director']

new_movies = movies[['movie_id','title','tags']]

new_movies['tags'] = new_movies['tags'].apply(lambda x : " ".join(x))

new_movies['tags'] = new_movies['tags'].apply(lambda x : x.lower())

#Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000,stop_words ='english')

vectors = cv.fit_transform(new_movies['tags']).toarray()

values = cv.get_feature_names_out()
for value in values:
    print(value)

#Stemming
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
  y = []
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

new_movies['tags'] = new_movies['tags'].apply(stem)

#Cosine Distances for finding similar movies
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

#Recommender Function
def recommend(movie):
  movie_index = new_movies[new_movies['title'] == movie].index[0]
  distances = similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)) , reverse = True , key = lambda x : x[1])[1 : 6]
  
  for i in movies_list:
    print(new_movies.iloc[i[0]].title)



pickle.dump(new_movies.to_dict(),open('movie_dict.pkl','wb'))

pickle.dump(similarity, open('sim.pkl','wb'))