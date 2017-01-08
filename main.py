# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import Imputer
import re
from sklearn import cross_validation
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes
from sklearn import tree
from sklearn import preprocessing
import matplotlib.pyplot as plt


movie_data  = pd.read_csv('movie_metadata.csv')
unique_genres = set()
overall_mean = 0

def calculate_mean():
    global movie_data
    global overall_mean
    movie_data = movie_data.dropna()
    overall_mean = round(movie_data['gross'].mean(), 0)
    gross_row = pd.Series(movie_data['gross'])
    gross_buckets = [100000, 1000000, 10000000, 100000000]
    gross_row[gross_row < gross_buckets[0]] = 5
    gross_row[gross_row > gross_buckets[3]] = 1
    gross_row[gross_row > gross_buckets[2]] = 2
    gross_row[gross_row > gross_buckets[1]] = 3
    gross_row[gross_row > gross_buckets[0]] = 4
    movie_data['gross'] = gross_row
            

#Normalise data
def normalise_data():
    global movie_data
    scalingObj = preprocessing.MinMaxScaler()
    normalised_data = scalingObj.fit_transform(movie_data)
    movie_data = normalised_data
    
def preprocess_movie_dataframe():
    global movie_data
    global unique_genres
    
    #Remove all movies not made in the US    
    movie_data = movie_data[movie_data['country'].str.contains("USA") == True]
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

    #Gather unique genres
    for index, row in movie_data.iterrows():
        genre  = re.sub('[|]', ' ', row['genres'])
        genre = genre.split() 
        unique_genres.add(genre[0])  
        '''
        If movie contains multilpe genres set genre of movie to be 
        first genre declared in list of genres
        '''
        movie_data.set_value(index, 'genres', genre[0])
        
    '''
    Sum total facebook actor likes with total director facebook likes to determine
    the hwo famous the cast of each movie is
    '''
    movie_data['cast_total_facebook_likes'] = movie_data['cast_total_facebook_likes'] + movie_data['director_facebook_likes']

    #Drop all unncecessary colunms
    del movie_data['plot_keywords']
    del movie_data['director_name']
    del movie_data['director_facebook_likes']
    del movie_data['actor_1_name']
    del movie_data['actor_1_facebook_likes']
    del movie_data['actor_2_name']
    del movie_data['actor_2_facebook_likes']
    del movie_data['actor_3_name']
    del movie_data['actor_3_facebook_likes']
    del movie_data['movie_title']
    
    '''
    Because we can't determine the total number of votes casted on a movie that
    hasn't been released we will drop the columns that contain these votes
    '''
    del movie_data['num_voted_users']
    del movie_data['num_critic_for_reviews']

def runClassifiers(data, target):
    
    dTree = tree.DecisionTreeClassifier()
    scores = cross_validation.cross_val_score(dTree, data, target, cv=10)
    print "Tree : ", scores.mean()
    
    rbfSvm = SVC()
    scores = cross_validation.cross_val_score(rbfSvm, data, target, cv=10)
    print "SVM : ", scores.mean()
    
    nearestN = KNeighborsClassifier()
    scores = cross_validation.cross_val_score(nearestN, data, target, cv=10)
    print "NNeighbour : ", scores.mean()
    
    randomForest = RandomForestClassifier()
    scores = cross_validation.cross_val_score(randomForest, data, target, cv=10)
    print "RForest : ",scores.mean()
    
    nBayes = naive_bayes.GaussianNB()
    scores = cross_validation.cross_val_score(nBayes, data, target, cv=10)
    print "Naive Bayes : ",scores.mean()
  
    
def atribute_mapping():
    global movie_data
    #Mapping for colour
    colour_mapping = {'Color':1, 'Black and White': 2}
    movie_data['color'] = movie_data['color'].map(colour_mapping)

    #Mapping Values for Content-Rating
    content_rating_mapping = {'G':1, 'PG':2, 'PG-13':3, 'R':4}
    movie_data['content_rating'] = movie_data['content_rating'].map(content_rating_mapping)

    #Perform one-hot encoding for country column
    movie_data = pd.get_dummies(movie_data, columns=["country"])
    
    #Perform one-hot encoding for language column
    movie_data = pd.get_dummies(movie_data, columns=["language"])

    #Perform one-hot encoding for genres column
    movie_data = pd.get_dummies(movie_data, columns=["genres"])   
    
def plot_graphs():
    sub_set = movie_data.head(1000)
    X = sub_set['cast_total_facebook_likes']
    Y = sub_set['gross']
    plt.xlabel("Cast Facebook Likes")
    plt.ylabel("Gross")
    plt.scatter(X, Y, marker="o", c=Y)
    plt.show()
    
    
  
def main():
    preprocess_movie_dataframe()
    atribute_mapping()
    plot_graphs()
    #calculate_mean()
    target = movie_data['gross']
    data = movie_data.drop(['gross'], axis=1)
    
    #runClassifiers(data, target)
    
    print "Normalising data..."
    #normalise_data()
    
    #runClassifiers(data, target)

    #

main()