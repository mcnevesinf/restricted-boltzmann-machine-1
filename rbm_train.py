
import numpy as np
import pandas as pd
import torch

from rbm import RBM


#Preparing the training set
data = pd.read_csv('ratings_small.csv')
train_data = data
#print(train_data.head())

#Getting the number of users and movies
nb_users = max(train_data['userId'])
nb_movies = max(train_data['movieId'])

print("Number of users: " + str(nb_users))
print("Number of movies: " + str(nb_movies))


#Convert the input data into an array with users in lines 
#and movies in columns
def convert(data):
	new_data = []
	
	for id_users in range(1, nb_users+1):
		id_movies = data['movieId'][data['userId'] == id_users]
		id_ratings = data['rating'][data['userId'] == id_users]
		
		ratings = np.zeros(nb_movies)
		ratings[id_movies - 1] = id_ratings
		
		new_data.append(list(ratings))

	return new_data
		
				
train_data = convert(train_data)

#Convert the data into Torch tensors
train_data = torch.FloatTensor(train_data)

#Convert the ratings into binary ratings
#1 (liked) or 0 (Not liked)
train_data[train_data == 0] = -1
train_data[train_data == 1] = 0
train_data[train_data == 2] = 0
train_data[train_data >= 3] = 1

#Set hyperparameters
nv = len(train_data[0]) #Number of visible nodes == number of movies
nh = 100
batch_size = 20
pre_trained = True

#Define RBM
rbm = RBM(nv, nh)

if pre_trained:
	rbm.load_rbm()
		
#Training the RBM model
print("Started training...")
nb_epoch = 10

for epoch in range(1, nb_epoch + 1):
	train_loss = 0
	s = 0.
	
	for id_user in range(0, nb_users - batch_size, batch_size):
		#Initialize next state of visible nodes
		vk = train_data[id_user:id_user+batch_size]
		
		#Input vector containing the movie ratings
		v0 = train_data[id_user:id_user+batch_size]

		#Vector of probabilities
		ph0,_ = rbm.sample_h(v0)
		
		#Sampling process
		for k in range(10):
			_,hk = rbm.sample_h(vk)
			_,vk = rbm.sample_v(hk)
			
			#Keep movies that are unseen and not 
			#recommended as such
			vk[v0<0] = v0[v0<0]
			
		phk,_ = rbm.sample_h(vk)
		
		rbm.train(v0, vk, ph0, phk)
		train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
		s += 1.
		
	print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

print("Training complete")


#Save trained model
print("Saving model...")
rbm.save_rbm()






















