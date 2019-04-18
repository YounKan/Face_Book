#!/usr/bin/env python
#
# Project: Video Streaming with Flask
# Author: Log0 <im [dot] ckieric [at] gmail [dot] com>
# Date: 2014/12/21
# Website: http://www.chioka.in/
# Description:
# Modified to support streaming out with webcams, and not just raw JPEGs.
# Most of the code credits to Miguel Grinberg, except that I made a small tweak. Thanks!
# Credits: http://blog.miguelgrinberg.com/post/video-streaming-with-flask
#
# Usage:
# 1. Install Python dependencies: cv2, flask. (wish that pip install works like a charm)
# 2. Run "python main.py".
# 3. Navigate the browser to the local webpage.
from flask import Flask, render_template, Response, redirect,url_for, jsonify
from camera import VideoCamera
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD, evaluate
import json

app = Flask(__name__)

cache = {}

databook = pd.read_json('booksdata.json')
# ratings = pd.read_csv('ratings.csv')

# tf =  TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
# tfidf_matrix = tf.fit_transform(databook['feature'])

# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

titles = databook[['title','authors','original_publication_year','language_code']]

# reader = Reader()
# data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
# svd = SVD()
# trainset = data.build_full_trainset()
# svd.train(trainset)

def find_feature(title):
  feature =  databook[(databook['title'] == title)].index[0]
  return feature

def get_recommendations(title):
    idx = find_feature(title)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]

def hybrid(userId):
  userbooks = ratings[(ratings['user_id'] == userId)]
  userbooks = userbooks.merge(databook[['title', 'book_id']], on='book_id')
  titlebook = list(userbooks['title'])
  
  bookrecs = pd.DataFrame()

  for x in titlebook:
      print(x)
      idx = find_feature(x)
      sim_scores = list(enumerate(cosine_sim[idx]))
      sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
      sim_scores = sim_scores[1:31]
      book_indices = [i[0] for i in sim_scores]
      bookrec = databook.iloc[book_indices][['book_id','title','authors','original_publication_year','language_code']]
      bookrecs = pd.concat([pd.DataFrame(bookrec), bookrecs], ignore_index=True)
      bookrecs['est'] = bookrecs['book_id'].apply(lambda x: svd.predict(userId, x).est)
      bookrecs = bookrecs.sort_values('est', ascending=False)
      
  return bookrecs.head(10)

@app.route('/')
def index():
    cache['facerec'] = True
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if camera.count == 10:
            cache['facerec'] = True
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live-data')
def live_data():
    print('live stream')
    def live_stream():
        while True:    
            if cache['facerec']:
                cache['facerec'] = False
                book_list = titles.head(5)

                bookjson = []
                for index, row in book_list.iterrows():
                    data ={"title": row['title'],
                           "authors": row['authors']}
                    bookjson.append(data)
                
                # book_list = hybrid(int(2)).to_json
                yield "data: " + json.dumps(bookjson) + "\n\n"
    return Response(live_stream(), mimetype= 'text/event-stream')

@app.route('/s')
def Home():
    return "<h1>Hello World!</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)