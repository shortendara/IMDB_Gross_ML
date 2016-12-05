# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import Imputer
import re


movie_data  = pd.read_csv('movie_metadata.csv')
mean_gross = 0

def calculate_mean_gross():
    global mean_gross
    mean_gross = movie_data['gross'].sum() / len(movie_data)
    print mean_gross

def calculate_movie_profitable():
    '''
    Compare budget and gross create colunm with binary values
    if the movie made a profit or not. Use new colunm as target 
    colunm
    ''' 
    rows_id = movie_data[movie_data['gross'] > movie_data['budget']].index


def preprocess_movie_dataframe():
    global movie_data
    #Set null movie durations to mean
    duration_imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=1)
    duration_column = duration_imputer.fit_transform(movie_data['duration']).T
    movie_data['duration'] = duration_column
    
    #Drop rows without a gross value
    movie_data = movie_data[pd.notnull(movie_data['gross'])]
    
    #Delete IMDB link from frame
    del movie_data['movie_imdb_link']
    
    #Delete IMDB score
    del movie_data['imdb_score']
    
    #Delete face count in poster
    del movie_data['facenumber_in_poster']

    #Delete movie facebook likes
    del movie_data['movie_facebook_likes']

def atribute_mapping():
    #Mapping Values for Content-Rating
    content_rating_mapping = {'G':0, 'PG':1, 'PG-13':2, 'R':3}
    movie_data['content_rating'] = movie_data['content_rating'].map(content_rating_mapping)

    #Mapping values for countries
    countries = movie_data['country']
    countries = set(countries)
    country_mapping = {}

    country_counter = 1
    for country in countries:
        country_mapping[country] = country_counter
        country_counter += 1

    movie_data['country'] = movie_data['country'].map(country_mapping)

    #Mapping for languages
    languages = movie_data['language']
    languages = set(languages)
    language_mapping = {}

    language_counter = 1
    for language in languages:
        language_mapping[language] = language_counter
        language_counter += 1

    movie_data['language'] = movie_data['language'].map(language_mapping)

    #Mapping values for movie genre
    genres = movie_data['genres']
    total_genres = set(genres)
    unique_genres =set()
    genre_mapping = {}
    genre_counter = 1

    #Gather unique genres
    for genre in total_genres:
        genre  = re.sub('[|]', ' ', genre)
        genre = genre.split()
        #Must convert genre into set or remove duplicate values
        unique_genres.add(genre[0])

    #Assign each genre a numerical value
    for genre in unique_genres:
        #Access first genre type and use as default genre
        genre_mapping[genre] = genre_counter
        genre_counter += 1
    
    movie_data['genres'] = movie_data['genres'].map(genre_mapping)

    '''
    #Mapping values for Director
    director_like_mapping = {}
    print movie_data['director_facebook_likes']
    #print movie_data.head()
    #Mapping values for first Actor
    actor_like_mapping ={}

    #Mapping values for second Actor
    second_actor_mapping = {}

    #Mapping values for third Actor
    third_actor_mapping = {}

    #Mapping values for movie duration
    movie_diration_mapping = {}
    '''

   
preprocess_movie_dataframe()
calculate_mean_gross() 
calculate_movie_profitable()
atribute_mapping()
