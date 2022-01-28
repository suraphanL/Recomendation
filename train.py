import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from surprise import dump
import os

def main():
    model_filename = "./model/model.pickle"
    ratings_path = './data/ratings.csv'

    ratings_df = pd.read_csv(ratings_path)
    ratings_df = ratings_df.dropna()
    ratings_df['userId'] = ratings_df['userId'].astype('int64')
    ratings_df['movieId'] = ratings_df['movieId'].astype('int64')
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_df[['userId','movieId','rating']], reader)
    trainset = data.build_full_trainset()
    algorithm = SVD()
    algorithm.fit(trainset)
    file_name = os.path.expanduser(model_filename)
    dump.dump(file_name, algo=algorithm)
    print (">> Dump done")
    print(model_filename)

if __name__=="__main__":
    main()