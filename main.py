# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import Imputer
import re
from sklearn import cross_validation
from sklearn.svm import SVR


movie_data  = pd.read_csv('movie_metadata.csv')

def calculate_movie_profitable():
    #Row numbers where movie made profit i.e successful movies
    rows_id = movie_data[movie_data['gross'] > movie_data['budget']].index
    for index, colunm in movie_data.iterrows():
        if index in rows_id:
            movie_data['profitable'] = 2
        else:
            movie_data['profitable'] = 1
    del movie_data['gross']
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

    del movie_data['plot_keywords']

    del movie_data['genres']

    del movie_data['director_name']

    del movie_data['actor_1_name']

    del movie_data['actor_2_name']

    del movie_data['actor_3_name']

    del movie_data['movie_title']

    #Drop any row containing NaN
    movie_data.dropna()

def atribute_mapping():
    #Mapping for colour
    colour_mapping = {'Color':1, 'Black and White': 2}
    movie_data['color'] = movie_data['color'].map(colour_mapping)

    #Mapping Values for Content-Rating
    content_rating_mapping = {'G':1, 'PG':2, 'PG-13':3, 'R':4}
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

    '''
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

def calculate_probability():
    movie_data_array = movie_data.as_matrix()
    target = movie_data_array[:,15]
    data = movie_data_array[:,0:14]
    estimator = SVR(kernel="linear")

    scores = cross_validation.cross_val_score(estimator, target, data, cv=10)
    print "initial values:" + scores.mean()
   
preprocess_movie_dataframe()
calculate_movie_profitable()
atribute_mapping()
calculate_probability()
print movie_data.head()
