import pandas as pd
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

# importing the dataset
df = pd.read_csv('InitialOutput.csv')

# Updating the Release_month field with the names of the corresponding months
months = []
for m in df['Release_month']:
    if m is 1:
        months.append('January')
    elif m is 2:
        months.append('February')
    elif m is 3:
        months.append('March')
    elif m is 4:
        months.append('April')
    elif m is 5:
        months.append('May')
    elif m is 6:
        months.append('June')
    elif m is 7:
        months.append('July')
    elif m is 8:
        months.append('August')
    elif m is 9:
        months.append('September')
    elif m is 10:
        months.append('October')
    elif m is 11:
        months.append('November')
    elif m is 12:
        months.append('December')

df['Release_month'] = months

df.to_csv('test.csv',index=False)

df_1 = pd.read_csv('test.csv')

# Splitting the Release_month column into distinct columns for each month 
#list_months = ['January','February','March','April','May','June','July','August','September','October','November','December']
list_months = []
for m in df_1['Release_month']:
    if m is not list_months:
        list_months.append(m)
    else:
        continue
    
for month in list_months:
    df_1[month] = df_1['Release_month'].str.contains(month).apply(lambda x:1 if x else 0)

# categorizing the popularity of the movies 
def popularityCategory(p):
    if p <= 5:
        return 0
    elif p <= 10:
        return 1
    elif p <= 20:
        return 2
    elif p <= 35:
        return 3
    elif p <= 50:
        return 4
    elif p <= 150:
        return 5
    elif p > 150:
        return 6
    
df_1['popularity'] = df_1['popularity'].apply(popularityCategory)

# categorizing the propularity of the actors
def actor1Category(a1):
    if a1 <= 2:
        return 0
    elif a1 <= 4:
        return 1
    elif a1 <= 6:
        return 2
    elif a1 <= 10:
        return 3
    elif a1 <= 20:
        return 4
    elif a1 > 20:
        return 5
    
df_1['Popularity_Actor 1'] = df_1['Popularity_Actor 1'].apply(actor1Category)

def actor2Category(a2):
    if a2 <= 1:
        return 0
    elif a2 <= 3:
        return 1
    elif a2 <= 5:
        return 2
    elif a2 <= 8:
        return 3
    elif a2 <= 14:
        return 4
    elif a2 > 14:
        return 5
        
df_1['Popularity_Actor 2'] = df_1['Popularity_Actor 2'].apply(actor2Category)

def actor3Category(a3):
    if a3 <= 1:
        return 0
    elif a3 <= 3:
        return 1
    elif a3 <= 5:
        return 2
    elif a3 <= 8:
        return 3
    elif a3 <= 14:
        return 4
    elif a3 > 14:
        return 5
      
df_1['Popularity_Actor 3'] = df_1['Popularity_Actor 3'].apply(actor3Category)

# Scaling the revenue values between 0 & 1
revValues = df_1['revenue'].values #returns a numpy array
revValuesTo2D = revValues.reshape(1,-1) ##converts it to 2D array
revNormalized = preprocessing.normalize(revValuesTo2D, norm='l2')
revNormalized = revNormalized.reshape(-1,1) ###convert it back to 1D array
df_1['revenue'] = revNormalized

# categorizing the revenue of the movies
def revenueCategory(revenue):
    if revenue <= 0:
        return 0
    elif revenue <= 0.0015:
        return 1
    elif revenue <= 0.008:
        return 2
    elif revenue > 0.008:
        return 3

df_1['revenue'] = df_1['revenue'].apply(revenueCategory)
    
# Filling the empty row values & Scaling the movie runtime values
df_1['runtime'] = df_1['runtime'].fillna(df_1['runtime'].mean())
x = df_1['runtime'].values 
x = x.reshape(1,-1)
x_normalized = preprocessing.normalize(x, norm='l2')
x_reshaped = x_normalized.reshape(-1,1) 
df_1['runtime'] = x_reshaped

# categorizing the runtime of the movie
def time(t):
    if t <= 0.012:
        return 0
    elif t <= 0.013:
        return 1
    elif t <= 0.014:
        return 2
    elif t <= 0.016:
        return 3
    elif t > 0.016:
        return 4
        
df_1['runtime'] = df_1['runtime'].apply(time)

# Filling the empty row values & Scaling the movie budget
df_1['budget'] = df_1['budget'].fillna(df_1['budget'].mean())
budgetValues = df_1['budget'].values 
budgetValuesTo2D = budgetValues.reshape(1,-1) 
budgetNormalized = preprocessing.normalize(budgetValuesTo2D, norm='l2')
budgetNormalized = budgetNormalized.reshape(-1,1)
df_1['budget'] = budgetNormalized

# categorizing the budget of the movie
def budgetCategory(budget):
    if budget <= 0:
        return 0
    elif budget <= 0.004:
        return 1
    elif budget <= 0.015:
        return 2
    elif budget > 0.015:
        return 3

df_1['budget'] = df_1['budget'].apply(budgetCategory)

# Filling the empty row values & Scaling the votes count
df_1['vote_count'] = df_1['vote_count'].fillna(df_1['vote_count'].mean())
voteCountValues = df_1['vote_count'].values 
voteCountValuesTo2D = voteCountValues.reshape(1,-1) 
voteCountNormalized = preprocessing.normalize(voteCountValuesTo2D, norm='l2')
voteCountNormalized = voteCountNormalized.reshape(-1,1) 
df_1['vote_count'] = voteCountNormalized

# categorizing the votes count for the movie
def votecount(count):
    if count <= 0.0005:
        return 0
    elif count <= 0.0025:
        return 1
    elif count <= 0.005:
        return 2
    elif count <= 0.02:
        return 3
    elif count > 0.02:
        return 4
       
df_1['vote_count'] = df_1['vote_count'].apply(votecount)

# Scaling the popularity of the directors
director = df_1['Popularity_Director'].values #returns a numpy array
directorTo2D = director.reshape(1,-1) ##converts to 2D array
directorNormalized = preprocessing.normalize(directorTo2D, norm='l2')
directorNormalized = directorNormalized.reshape(-1,1) ###convert back to 1D array
df_1['Popularity_Director'] = directorNormalized

# categorizing the popularity of the directors
def directorCategory(d):
    if d <= 0:
        return 0
    elif d <= 0.0005:
        return 1
    elif d <= 0.007:
        return 2
    elif d <= 0.015:
        return 3
    elif d > 0.015:
        return 4
    
df_1['Popularity_Director'] = df_1['Popularity_Director'].apply(directorCategory)

# Scaling the popularity of the director of photography
dop = df_1['Popularity_DOP'].values 
dopTo2D = dop.reshape(1,-1) 
dopNormalized = preprocessing.normalize(dopTo2D, norm='l2')
dopNormalized = dopNormalized.reshape(-1,1) 
df_1['Popularity_DOP'] = dopNormalized

# categorizing the popularity of the director of photography
def dopCategory(dop):
    if dop <= 0:
        return 0
    elif dop <= 0.00001:
        return 1
    elif dop <= 0.001:
        return 2
    elif dop > 0.001:
        return 3
      
df_1['Popularity_DOP'] = df_1['Popularity_DOP'].apply(dopCategory)

# Scaling the popularity of the screenplay head
screenplay = df_1['Popularity_Screenplay'].values 
screenplayTo2D = screenplay.reshape(1,-1) 
screenplayNormalized = preprocessing.normalize(screenplayTo2D, norm='l2')
screenplayNormalized = screenplayNormalized.reshape(-1,1) 
df_1['Popularity_Screenplay'] = screenplayNormalized

# Scaling the popularity of the music composer
mc = df_1['Popularity_MusicComposer'].values 
mcTo2D = mc.reshape(1,-1) 
mcNormalized = preprocessing.normalize(mcTo2D, norm='l2')
mcNormalized = mcNormalized.reshape(-1,1) 
df_1['Popularity_MusicComposer'] = mcNormalized

# Scaling the popularity of the producer
producer = df_1['Popularity_Producer'].values 
producerTo2D = producer.reshape(1,-1) 
producerNormalized = preprocessing.normalize(producerTo2D, norm='l2')
producerNormalized = producerNormalized.reshape(-1,1) 
df_1['Popularity_Producer'] = producerNormalized

# categorizing the popularity of the producer
def producerCategory(producer):
    if producer <= 0:
        return 0
    elif producer <= 0.000007:
        return 1
    elif producer <= 0.0005:
        return 2
    elif producer <= 0.007:
        return 3
    elif producer > 0.007:
        return 4
        
df_1['Popularity_Producer'] = df_1['Popularity_Producer'].apply(producerCategory)

# elimating few columns
cols = ['movie_id','title','Popularity_MusicComposer','Popularity_Screenplay','Fantasy','Documentary','Family','Science Fiction','Music','History','Animation','Mystery','Western','War','TV Movie','Foreign','Horror','Popularity_Actor 4','Actor 1','Actor 2','Actor 3','Actor 4','Director','DOP','Writer','Screenplay','Music Composer','Stunts Director','Producer','production_companies','vote_average','Release_month']
df_1.drop(cols, axis = 1, inplace = True)

df_1.to_csv('FinalOutput.csv',index=False)

# splitting the data into training and testing set based on the movie release year
traindatasplit = df_1[df_1['Release_year'] <= 2013]
testdatasplit = df_1[df_1['Release_year'] > 2013]

traindatasplit.to_csv('trainData.csv',index=False)
testdatasplit.to_csv('testData.csv',index=False)



