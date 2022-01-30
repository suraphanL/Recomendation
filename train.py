import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import dump
import os

model_filename = "./model/model.pickle"
ratings_path = './data/ratings.csv'

def main():
    ratings_df = pd.read_csv(ratings_path)
    ratings_df = ratings_df.dropna()
    ratings_df['userId'] = ratings_df['userId'].astype('int64')
    ratings_df['movieId'] = ratings_df['movieId'].astype('int64')
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_df[['userId','movieId','rating']], reader)
    trainset = data.build_full_trainset()
    
    param_grid = {'n_epochs': [20, 30], 'lr_all': [0.005, 0.010],
                'n_factors': [50, 100]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    print("Best RMSE score: ", gs.best_score['rmse'])
    print(gs.best_params['rmse'])
    params = gs.best_params['rmse']
    algorithm = SVD(n_epochs = params['n_epochs'], lr_all = params['lr_all'], n_factors = params['n_factors'])
    algorithm.fit(trainset)
    file_name = os.path.expanduser(model_filename)
    dump.dump(file_name, algo=algorithm)
    print (">> Dump done")
    print(model_filename)

def load_model(model_filename):
    print (">> Loading dump")
    from surprise import dump
    import os
    file_name = os.path.expanduser(model_filename)
    _, loaded_model = dump.load(file_name)
    print (">> Loaded dump")
    return loaded_model

def predict_with_loaded_model():
    ratings_df = pd.read_csv(ratings_path)
    ratings_df = ratings_df.dropna()
    ratings_df['userId'] = ratings_df['userId'].astype('int64')
    ratings_df['movieId'] = ratings_df['movieId'].astype('int64')
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_df[['userId','movieId','rating']], reader)
    trainset = data.build_full_trainset()
    loaded_model = load_model(model_filename)
    print("Model done")
    predictions = loaded_model.test(trainset.build_anti_testset())
    print(predictions[:10])

if __name__=="__main__":
    main()
    # predict_with_loaded_model()