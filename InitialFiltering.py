import pandas as pd
import json
import csv
import matplotlib.pyplot as plt
import numpy as np

movies_data = pd.read_csv('tmdb_5000_credits.csv')
credits_data = pd.read_csv('tmdb_5000_movies.csv')

# Combining the movie database data into a single csv file
output = pd.merge(movies_data,credits_data)
output.to_csv('InitialDataSet.csv')

df = pd.read_csv('InitialDataSet.csv')

# Dataset with json columns
json_columns = ['cast','crew','genres','production_companies']

for column in json_columns:
    df[column] = df[column].apply(json.loads)

df_1 = df[['movie_id','title','budget','runtime','popularity','revenue','vote_average','vote_count','production_companies']].reset_index(drop=True)

df_1['runtime'] = df_1['runtime'].fillna(df_1['runtime'].mean())

def flatten_names(keywords):
    return '|'.join([x['name'] for x in keywords])

# Extracting all the genres of a movie
df['genres'] = df['genres'].apply(flatten_names)

liste_genres = set()
for s in df['genres'].str.split('|'):
    liste_genres = set().union(s, liste_genres)
liste_genres = list(liste_genres)
liste_genres.remove('')

# Splliting the genres into separate columns
for genre in liste_genres:
    df_1[genre] = df['genres'].str.contains(genre).apply(lambda x:1 if x else 0)

def retreive_data(data, positions):
    result = data
    try:
        for id in positions:
            result = result[id]
        return result
    except IndexError or KeyError:
        return pd.np.nan

# Extracting the actors(cast) data from the json columns
df_1['Actor 1'] = df['cast'].apply(lambda x: retreive_data(x, [0, 'id']))
df_1['Actor 2'] = df['cast'].apply(lambda x: retreive_data(x, [1, 'id']))
df_1['Actor 3'] = df['cast'].apply(lambda x: retreive_data(x, [2, 'id']))
df_1['Actor 4'] = df['cast'].apply(lambda x: retreive_data(x, [3, 'id']))

# Filling the missing values in the columns
df_1['Actor 1'] = df_1['Actor 1'].fillna('0')
df_1['Actor 2'] = df_1['Actor 2'].fillna('0')
df_1['Actor 3'] = df_1['Actor 3'].fillna('0')
df_1['Actor 4'] = df_1['Actor 4'].fillna('0')

# Extracting the names of directors from the crew column and filling the missing values in the column
def director_name(crew_data):
    directors = [x['id'] for x in crew_data if x['job'] == 'Director']
    return retreive_data(directors, [0])

df_1['Director'] = df['crew'].apply(director_name)
df_1['Director'] = df_1['Director'].fillna('0')

# Extracting the names of director of photography from the crew column and filling the missing values in the column
def dop_name(crew_data):
    dop = [x['id'] for x in crew_data if x['job'] == 'Director of Photography']
    return retreive_data(dop, [0])

df_1['DOP'] = df['crew'].apply(dop_name)
df_1['DOP'] = df_1['DOP'].fillna('0')

# Extracting the name of writer from the crew column and filling the missing values in the column
def writer_name(crew_data):
    writer = [x['id'] for x in crew_data if x['job'] == 'Writer']
    return retreive_data(writer, [0])

df_1['Writer'] = df['crew'].apply(writer_name)
df_1['Writer'] = df_1['Writer'].fillna('0')

# Extracting the names of screenplay head from the crew column and filling the missing values in the column
def screenplay(crew_data):
    screenplay = [x['id'] for x in crew_data if x['job'] == 'Screenplay']
    return retreive_data(screenplay, [0])

df_1['Screenplay'] = df['crew'].apply(screenplay)
df_1['Screenplay'] = df_1['Screenplay'].fillna('0')

# Extracting the names of music composers from the crew column and filling the missing values in the column
def music_composer_name(crew_data):
    music_composer = [x['id'] for x in crew_data if x['job'] == 'Original Music Composer']
    return retreive_data(music_composer, [0])

df_1['Music Composer'] = df['crew'].apply(music_composer_name)
df_1['Music Composer'] = df_1['Music Composer'].fillna('0')

# Extracting the names of stuntman from the crew column and filling the missing values in the column
def stunts_name(crew_data):
    stunts = [x['id'] for x in crew_data if x['job'] == 'Stunts']
    return retreive_data(stunts, [0])

df_1['Stunts Director'] = df['crew'].apply(stunts_name)
df_1['Stunts Director'] = df_1['Stunts Director'].fillna('0')

# Extracting the names of producers from the crew column and filling the missing values in the column
def producer_name(crew_data):
    producer = [x['id'] for x in crew_data if x['job'] == 'Producer']
    return retreive_data(producer, [0])

df_1['Producer'] = df['crew'].apply(producer_name)
df_1['Producer'] = df_1['Producer'].fillna('0')

# Extracting the names of production companies from the crew column and filling the missing values in the column
def production_company_name(production_data):
    pro = [x['id'] for x in production_data]
    return retreive_data(pro, [0])

df_1['production_companies'] = df['production_companies'].apply(production_company_name)
df_1['production_companies'] = df_1['production_companies'].fillna('0')

# Extracting the release year and month of a movie
from datetime import datetime

dt = df['release_date']
data = pd.to_datetime(dt)
month = data.dt.month
df_1['Release_month'] = month
year = data.dt.year
df_1['Release_year'] = year

df1 = pd.DataFrame(df_1)
df1.to_csv("output.csv",sep=',',index=False)

# Retreiving the popularity information of the cast and crew data from the TMDB API
import tmdbsimple as tmdb
import json
tmdb.API_KEY = '02d4d7373cb76210bc18a4a0912c0f31'
popularity_actor1 = []
popularity_actor2 = []
popularity_actor3 = []
popularity_actor4 = []
director = []
dop = []
screenplay = []
music_composer = []
producer = []

for i in df_1['Actor 1']:
    try:
        movie = tmdb.People(i)
        response = movie.info()
        popularity_actor1.append(response['popularity'] )
    except:
        popularity_actor1.append('0')

df_1['Popularity_Actor 1'] = popularity_actor1

for j in df_1['Actor 2']:
    try:
        movie = tmdb.People(j)
        response = movie.info()
        popularity_actor2.append(response['popularity'])
    except:
        popularity_actor2.append('0')
            
df_1['Popularity_Actor 2'] = popularity_actor2

for k in df_1['Actor 3']:
    try:
        movie = tmdb.People(k)
        response = movie.info()
        popularity_actor3.append(response['popularity'] )
    except:
        popularity_actor3.append('0')
    
df_1['Popularity_Actor 3'] = popularity_actor3

for m in df_1['Actor 4']:
    try:
        movie = tmdb.People(m)
        response = movie.info()
        popularity_actor4.append(response['popularity'] )
    except:
        popularity_actor4.append('0')

df_1['Popularity_Actor 4'] = popularity_actor4

for n in df_1['Director']:
    try:
        movie = tmdb.People(n)
        response = movie.info()
        director.append(response['popularity'] )
    except:
        director.append('0')

df_1['Popularity_Director'] = director

for o in df_1['DOP']:
    try:
        movie = tmdb.People(o)
        response = movie.info()
        dop.append(response['popularity'] )
    except:
        dop.append('0')

df_1['Popularity_DOP'] = dop

for p in df_1['Screenplay']:
    try:
        movie = tmdb.People(p)
        response = movie.info()
        screenplay.append(response['popularity'] )
    except:
        screenplay.append('0')

df_1['Popularity_Screenplay'] = screenplay

for q in df_1['Music Composer']:
    try:
        movie = tmdb.People(q)
        response = movie.info()
        music_composer.append(response['popularity'] )
    except:
        music_composer.append('0')

df_1['Popularity_MusicComposer'] = music_composer

for r in df_1['Producer']:
    try:
        movie = tmdb.People(r)
        response = movie.info()
        producer.append(response['popularity'] )
    except:
        producer.append('0')

df_1['Popularity_Producer'] = producer

df_1.to_csv('InitialDataSet.csv',index=False)


