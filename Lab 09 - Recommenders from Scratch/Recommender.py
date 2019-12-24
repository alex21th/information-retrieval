'''

Program that reproduces a recommender system using a KNN approach.


__authors__ = David Berges Llado and Alex Carrillo Alza

'''


import csv
import argparse
import numpy as np

from collections import OrderedDict

"""implements a recommender system built from
   a movie list name
   a listing of userid+movieid+rating"""
class Recommender():

    #"""initializes a recommender from a movie file and a ratings file"""
    def __init__(self, movie_filename, rating_filename):

        # read movie file and create dictionary _movie_names
        self._movie_names = {}
        with open(movie_filename, 'r', encoding = 'utf8') as csv_reader:
            reader = csv.reader(csv_reader)
            next(reader, None)

            for line in reader:
                movieid = line[0]
                moviename = line[1]
                # ignore line[2], genre
                self._movie_names[movieid] = moviename

        # read rating file and create _movie_ratings (ratings for a movie)
        # and _user_ratings (ratings by a user) dicts
        self._movie_ratings = {}
        self._user_ratings = {}

        with open(rating_filename, 'r', encoding = 'utf8') as csv_reader:
            reader = csv.reader(csv_reader)
            next(reader, None)

            for line in reader:

                userid = line[0]
                movieid = line[1]
                rating = line[2]
                # ignore line[3], timestamp
                if userid in self._user_ratings:
                    userrats = self._user_ratings[userid]
                else:
                    userrats = {}
                userrats.update({movieid: float(rating)})
                self._user_ratings[userid] = userrats

                if movieid in self._movie_ratings:
                    movierats = self._movie_ratings[movieid]
                else:
                    movierats = {}
                movierats.update({userid: float(rating)})
                self._movie_ratings[movieid] = movierats


    ####
    # USER TO USER
    ####

    
    def similarity_between_users(self, ratings1, ratings2):
        ''' Function that computes the similarity between users
        ----------
        PARAMETERS
        - ratings1, ratings2: two dictionaries representing the rating lists of two users
        ----------
        RETURNS
        - float representing the similarity

        '''
        avg1 = np.array(list(ratings1.values())).mean()
        avg2 = np.array(list(ratings2.values())).mean()

        S = set(ratings1.keys()).intersection(set(ratings2.keys()))

        num, den1, den2 = 0., 0., 0.
        for movieid in S:
            num += (ratings1[movieid] - avg1)*(ratings2[movieid] - avg2)
            den1 += (ratings1[movieid] - avg1)**2
            den2 += (ratings2[movieid] - avg2)**2

        # return round(num / (den1*den2), 5) if (den1 != 0 and den2 != 0) else 0
        return num / np.sqrt(den1*den2) if (den1 != 0 and den2 != 0) else 0

    
    def predict_rating(self, movie, neighbours):
        ''' Function that predicts the rating of a movie for a given user
        ----------
        PARAMETERS
        - movie: movieId of the movie we want to predict its rating
        - neighbours: list of the 'knn' nearest neighbours to the user in question
        ----------
        RETURNS
        - float between 1-5 representing the rating for that specific movie

        '''
        num, den = 0., 0.
        for neighbour, similarity in neighbours.items():
            # Check that the movie has been seen by this specific user
            if movie in self._user_ratings[neighbour]:
                # We only want to consider significant positive similarities
                if similarity >= .01:
                    num += similarity * (self._user_ratings[neighbour][movie])
                    den += similarity

        return num/den if (den != 0) else 0


    def recommend_user_to_user(self, rating_list, knn = 50, k = 10):
        ''' Function that returns the 'k' most likely movies for a specific user to like
        ----------
        PARAMETERS
        - rating_list: dictionary representing a rating list for a new user
        - knn: integer representing the number of nearest neighbours to take into account
        - k: integer representing the number of recommendations to get
        ----------
        RETURNS
        - a dictionary with the 'k' highest recommended movies to watch for the user

        '''
        neighbours = {}
        for user, ratings in self._user_ratings.items():
            neighbours[user] = self.similarity_between_users(rating_list, ratings)
        # Sort the dictionary
        neighbours = sorted(neighbours.items(), key = lambda x: -x[1])
        # Stick with the closest 'knn' users
        neighbours = OrderedDict(neighbours[:knn])

        # Get the movies that have been reviewed by some of the knn neighbours and that
        # the user has not reviewed.
        movies_to_rate = set()
        for user in neighbours.keys():
            movies_to_rate = movies_to_rate.union(set(self._user_ratings[user].keys()))
        movies_to_rate = [x for x in movies_to_rate if x not in rating_list.keys()]

        # Predict the rating that our user would rate for each movie in 'movies_to_rate'
        pred = {}
        for movie in movies_to_rate:
            pred[movie] = self.predict_rating(movie, neighbours)
        # Sort the dictionary
        pred = sorted(pred.items(), key = lambda x: -x[1])
        # Stick with the first k movies
        pred = OrderedDict(pred[:k])

        return pred


    ####
    # ITEM TO ITEM
    ####

    def similarity_between_items(self, ratings1, ratings2):
        ''' Function that computes the similarity between items (movies)
        ----------
        PARAMETERS
        - ratings1, ratings2: two dictionaries representing the rating lists of two users
        ----------
        RETURNS
        - float representing the similarity

        '''
        avg1 = np.array(list(ratings1.values())).mean()
        avg2 = np.array(list(ratings2.values())).mean()

        S = set(ratings1.keys()).intersection(set(ratings2.keys()))

        num, den1, den2 = 0., 0., 0.
        for userid in S:
            num += (ratings1[userid] - avg1)*(ratings2[userid] - avg2)
            den1 += (ratings1[userid] - avg1)**2
            den2 += (ratings2[userid] - avg2)**2

        return num / np.sqrt(den1*den2) if (den1 != 0 and den2 != 0) else 0


    def predict_rating2(self, rating_list, neighbours):
        ''' Function that predicts the rating of a movie for a given user
        ----------
        PARAMETERS
        - rating_list: dictionary representing the rating list of the specific user
        - neighbours: list of the 'knn' nearest neighbours to the user in question
        ----------
        RETURNS
        - float between 1-5 representing the rating for that specific movie

        '''
        num, den = 0., 0.
        for movie, similarity in neighbours.items():
            # We only want to consider significant positive similarities
            if similarity >= .01:
                num += similarity * rating_list[movie]
                den += similarity

        return num / den if (den != 0) else 0


    def recommend_item_to_item(self, rating_list, knn, k):
        ''' Function that returns the 'k' most likely movies for a specific user to like
        ----------
        PARAMETERS
        - rating_list: dictionary representing a rating list for a new user
        - knn: integer representing the number of nearest neighbours to take into account
        - k: integer representing the number of recommendations to get
        ----------
        RETURNS
        - a dictionary with the 'k' highest recommended movies to watch for the user

        '''
        # Get the movies that the user has not rated
        movies_to_rate = [x for x in self._movie_ratings.keys() if x not in rating_list.keys()]

        # Get the closest movies
        neighbours = {}
        for movie1 in movies_to_rate:
            ratings = self._movie_ratings[movie1]
            neighbours[movie1] = {}
            for movie2 in rating_list.keys():
                neighbours[movie1][movie2] = self.similarity_between_items(ratings, self._movie_ratings[movie2])

        # Sort the dictionaries and select the first knn appearances
        for movie in neighbours:
            neighbours[movie] = sorted(neighbours[movie].items(), key = lambda x: -x[1])
            neighbours[movie] = OrderedDict(neighbours[movie][:knn])
        
        # Predict the rating for every possible movie
        pred = {}
        for movie in neighbours:
            pred[movie] = self.predict_rating2(rating_list, neighbours[movie])
        # Sort the movies by rating and stick with the k first
        pred = sorted(pred.items(), key = lambda x: -x[1])
        pred = OrderedDict(pred[:k])

        return pred


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-knn', default = 50, type = int, help = 'Number of closest neighbours to consider.'
    )
    parser.add_argument(
        '-k', default = 10, type = int, help = 'Number of objects shown to the user.'
    )
    # Get the arguments
    args = parser.parse_args()
    knn = args.knn
    k = args.k

    # Create the class reading the files
    r = Recommender("./ml-latest-small/movies.csv","./ml-latest-small/ratings.csv")

    # Repeatedly, asks for a list of movies and ratings, and asks the Recommender to provide
    # recommendations given this list and prints the titles of the recommended movies and their
    # predicted rating:
    while (input('New list of movie ratings? [y/n] : ') == 'y'):
        print('\nEnd the list by typing 0 in the movieID\n')
        rating_list = {}
        movie = input('MovieID : ')
        while (movie != '0'):
            assert (float(movie) <= 193609) & (float(movie) >= 1), f"MovieID {movie} doesn't exist. Ensure it is in range(1,193609)"
            # Input a rating and ensure it is valid
            rating = float(input('Rating : '))
            assert (rating <= 5.) & (rating >= .5), 'Rating range must be between 1 and 5'
            # Append the rating to the rating list and ask for a new movieID
            rating_list[movie] = rating
            movie = input('MovieID : ')
        print()

        # Predict the ratings using a User-to-User recommender
        print('-' * 60)
        print('Using User-to-User recommendation:')
        recommended = r.recommend_user_to_user(rating_list, knn, k)
        for movieid, rate in recommended.items():
            print(f" - {r._movie_names[movieid]} : {rate}")
        print()

        print('-' * 60)
        print('using Item-to-Item recommendation:')
        recommended = r.recommend_item_to_item(rating_list, knn, k)
        for movieid, rate in recommended.items():
            print(f" - {r._movie_names[movieid]} : {rate}")
        print()
        









