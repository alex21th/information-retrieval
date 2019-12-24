from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import Trainset
from surprise.model_selection import cross_validate
from surprise import KNNWithMeans
import os
import pandas as pd

######## RATINGS.CSV

# path to dataset file
file_path = os.path.expanduser('ratings.csv')

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format='user item rating timestamp', sep=',')

data = Dataset.load_from_file(file_path, reader=reader)

trainset = data.build_full_trainset()

####### MOVIES.CSV

movies = pd.read_csv("movies.csv", header = 0)

Knn = input("Algorithm fit with KNNwithMean (type knn) or SVD decomposition (type svd)?\n")

# Build an algorithm, and train it.
if Knn == "knn":
	algo = KNNWithMeans(biased = True)
	algo.fit(trainset)
elif Knn == "svd":
	algo = SVD(n_factors = 10, biased = True)
	algo.fit(trainset)
else:
	print("No algorithm found")

if Knn == "knn" or Knn == "svd":

	# Concatenate a user id and a movie id to predict a rating.
	iuid = input("iuid:\n")
	iiid = input("iiid:\n")

	uid = str(iuid)  # raw user id (as in the ratings file). They are **strings**!
	iid = str(iiid)  # raw item id (as in the ratings file). They are **strings**!

	# DON'T EXECUTE!
	# cross_validate(algo, data, verbose = True)

	# Get a prediction for specific users and items.
	pred = algo.predict(uid, iid, verbose=True)

	if Knn == "svd":
		k = input("k=? || Will show top k movies for its relevance to each latent factor")
		i = 0
		while i < len(algo.qi[0]):
			moviesLF = []
			j = 0
			while j < len(algo.qi):
				moviesLF.append((movies['title'].loc[movies['movieId'] == int(trainset.to_raw_iid(j))],algo.qi[j][i]))
				j+=1

			moviesLF.sort(key=lambda x: x[1],reverse=True)
			print(str(moviesLF[:int(k)])+"\n")
			i+=1
