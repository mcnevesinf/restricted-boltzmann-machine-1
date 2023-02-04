
import pandas as pd
import torch

from rbm import RBM

#Use train data to discover number of movies
data = pd.read_csv('ratings_small.csv')
train_data = data

nb_movies = max(train_data['movieId'])

#Load trained model
rbm = RBM()
rbm.load_rbm()

#Example input vector containing some movie ratings
#Set high ratings for animation movies as a test case
v = torch.ones(1, nb_movies) * (-1.)
v[0,1] = 0.0
v[0,2] = 0.0
v[0,11] = 1.0
v[0,23] = 0.0
v[0,97] = 0.0
v[0,862] = 1.0
v[0,7442] = 1.0
v[0,10136] = 1.0
v[0,10500] = 1.0
v[0,11429] = 1.0
v[0,13059] = 1.0
v[0,15654] = 1.0
v[0,17710] = 1.0
v[0,20454] = 1.0
v[0,49947] = 1.0

print("Number of movies: " + str( len(v[0]) ))

#Perform the inference
_,h = rbm.sample_h(v)
_,v = rbm.sample_v(h)
				
print("Number of recommended movies: " + str( len(v[v>0]) ))

#Load movie data to check a few recommended movies
movies = pd.read_csv('movies_metadata.csv', usecols=['id', 'original_title'])
movie_data = movies
#print(movie_data.head(5))

#Sanitize the dataset
for b in movie_data['id']:
	if '-' in b:
		movie_data[movie_data['id'] == b] = -1

movie_data['id'] = movie_data['id'].astype('int')

#Print ID and name of recommended movies
i = 0
j = 0

while i < 50:
	if v[0, j] == 1.0:
		df = movie_data.loc[movie_data['id'] == j+1]
		
		#If there is a movie with the recommended ID (i.e., j+1) 
		if not df.empty:
			selected_movie = movie_data[movie_data['id'] == j+1]
			print(selected_movie['id'].to_string(index=False) + '\t' +
			      selected_movie['original_title'].to_string(index=False))
			i += 1
	j += 1

	

















