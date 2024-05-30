import streamlit as st
import pandas as pd
import pickle
import requests

st.title('Movie Recommendation System')

def fetch_poster(movie_id):
   response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=b7c1bcae26072dd54fc95b4eb885e040&append_to_response=videos,images'.format(movie_id))
   data = response.json()
   return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

#Recommender Function
def recommend(movie):
  movie_index = movies[movies['title'] == movie].index[0]
  distances = similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)) , reverse = True , key = lambda x : x[1])[1 : 11]
  recommended_movies = []
  recommended_movies_posters = []
  for i in movies_list:
    movie_id = movies.iloc[i[0]].movie_id
    recommended_movies.append(movies.iloc[i[0]].title)
    recommended_movies_posters.append(fetch_poster(movie_id))

  return recommended_movies,recommended_movies_posters


movies_dict = pickle.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('sim.pkl','rb'))

selected_movie = st.selectbox(
    'Select a movie',
    movies['title'].values)

if st.button('Recommend'):
    names , posters = recommend(selected_movie)
    def display_movie_column(col, name, poster):
        with col:
            st.markdown(f'<div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; text-align : center" title="{name}">{name}</div>', unsafe_allow_html=True)
            st.image(poster)

    # Display first 5 columns in the same line
    col1, col2, col3, col4, col5 = st.columns(5)
    for i in range(5):
        display_movie_column(eval(f"col{i+1}"), names[i], posters[i])

    st.write(" ")
    st.write(" ")

    # Display the next 5 columns on the next line
    col6, col7, col8, col9, col10 = st.columns(5)
    for i in range(5, 10):
        display_movie_column(eval(f"col{i+1}"), names[i], posters[i])

    # selected_genre = st.sidebar.selectbox("Select Genre", genres)

    # # Filter movies by genre
    # filtered_names = [name for name, genre in zip(names, genres) if genre == selected_genre]
    # filtered_posters = [poster for poster, genre in zip(posters, genres) if genre == selected_genre]

   