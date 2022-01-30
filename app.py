from flask import Flask, redirect, url_for
from flask import request, jsonify
from collections import defaultdict
from surprise import Dataset, Reader
import pandas as pd
import os

def load_model(model_filename):
    print (">> Loading dump")
    from surprise import dump
    import os
    file_name = os.path.expanduser(model_filename)
    _, loaded_model = dump.load(file_name)
    print (">> Loaded dump")
    return loaded_model

app = Flask(__name__)
model_filename = "./model/model.pickle"
ratings_path = './data/ratings.csv'
movies_path = './data/movies.csv'

def get_top_n(predictions, n=10):
  top_n = defaultdict(list)
  for uid, mid, true_r, est, _ in predictions:
    top_n[uid].append((mid, est))
  for uid, user_ratings in top_n.items():
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    top_n[uid] = user_ratings[:n]
  return top_n

def calculate_top_rate_movies():
    average_ratings = pd.DataFrame(ratings_df.groupby('movieId')['rating'].mean())
    average_ratings['Total Ratings'] = pd.DataFrame(ratings_df.groupby('movieId')['rating'].count())
    average_ratings = average_ratings[average_ratings['Total Ratings']>100].sort_values('rating',ascending=False).reset_index()
    global top_rate_movies
    movie_ids = average_ratings['movieId'].tolist()
    top_rate_movies = list(map(lambda x: {"id": str(x)}, movie_ids))
    
    
@app.route('/')
def hello():
    return "<p>Hello, World!</p>"

@app.before_first_request
def before_first_request_func():
    print("This function will run once")
    global top_n
    global ratings_df
    global movies_df
    print(movies_path)        
    movies_df = pd.read_csv(movies_path)
    print(movies_df)
    ratings_df = pd.read_csv(ratings_path)
    ratings_df = ratings_df.dropna()
    ratings_df['userId'] = ratings_df['userId'].astype('int64')
    ratings_df['movieId'] = ratings_df['movieId'].astype('int64')
    print(ratings_df)
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_df[['userId','movieId','rating']], reader)
    trainset = data.build_full_trainset()
    print("Data Done")
    loaded_model = load_model(model_filename)
    print("Model done")
    predictions = loaded_model.test(trainset.build_anti_testset())
    print("predictions Done")
    top_n = get_top_n(predictions, n=10)
    calculate_top_rate_movies()
    print("Done")
    
@app.route('/recommendations', methods=['GET'])
def recommendations():
    print("recommendations")
    userId = int(request.args['user_id'])
    returnMetadata = request.args.get('returnMetadata', False, type=bool)
    print(returnMetadata)
    print("request received!")
    print(userId)
    movie_ids = list(map(lambda x: {"id": str(x[0])}, top_n[userId]))
    if len(movie_ids) == 0:
        print(len(movie_ids))
        movie_ids = top_rate_movies[:10]
        
    if returnMetadata:
        movies_detail = list(map(lambda x: movie_detail_by_id(int(x["id"])), movie_ids))
        data = {'items': movies_detail}
        print(data)
        return jsonify(data)
    else:
        data = {'items': movie_ids}
        print(data)
        return jsonify(data)

@app.route('/features', methods=['GET'])
def features():
    userId = int(request.args['user_id'])
    histories = get_feature_by_user_id(userId)
    data = {"features": [{"histories": histories}]}
    return jsonify(data)

def movie_detail_by_id(id):
    movie = movies_df[movies_df['movieId'] == id].iloc[0]
    detail = {"id": str(id), 
              "title": movie['title'], 
              "genres": movie['genres'].split("|")}
    return detail

def get_feature_by_user_id(id):
    movies = ratings_df[ratings_df['userId'] == id]['movieId'].astype('str').tolist()
    return movies

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)