from flask import Flask, render_template, Response, url_for
from camera import VideoCamera
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD, evaluate
import json
import math

app = Flask(__name__)

cache = {}

class DataStore():
    facerec = "None"

datalocal = DataStore()

databook = pd.read_json('booksdata.json')
ratings = pd.read_csv('ratings.csv')

# tf =  TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
# tfidf_matrix = tf.fit_transform(databook['feature'])

# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# titles = databook[['title','authors','small_image_url','average_rating']]

# reader = Reader()
# data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
# svd = SVD()
# trainset = data.build_full_trainset()
# svd.train(trainset)

# def find_feature(title):
#   feature =  databook[(databook['title'] == title)].index[0]
#   return feature

# def get_recommendations(title):
#     idx = find_feature(title)
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:31]
#     book_indices = [i[0] for i in sim_scores]
#     return titles.iloc[book_indices]

# def hybrid(userId):
#   userbooks = ratings[(ratings['user_id'] == userId)]
#   userbooks = userbooks.merge(databook[['title', 'book_id']], on='book_id')
#   titlebook = list(userbooks['title'])
  
#   bookrecs = pd.DataFrame()

#   for x in titlebook:
#       idx = find_feature(x)
#       sim_scores = list(enumerate(cosine_sim[idx]))
#       sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#       sim_scores = sim_scores[1:31]
#       book_indices = [i[0] for i in sim_scores]
#       bookrec = databook.iloc[book_indices][['book_id','title','authors','small_image_url','average_rating']]
#       bookrecs = pd.concat([pd.DataFrame(bookrec), bookrecs], ignore_index=True)
#       bookrecs['est'] = bookrecs['book_id'].apply(lambda x: svd.predict(userId, x).est)
#       bookrecs = bookrecs.sort_values('est', ascending=False)
      
#   return bookrecs.head(10)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if camera.count == 10:
            datalocal.facerec = camera.facerec
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
            if datalocal.facerec != None:
                datalocal.facerec = None
                # book_list = hybrid(int(2))
                book_list = databook.head(5)
                bookjson = []
                for index, row in book_list.iterrows():
                    data ={"title": row['title'],
                           "authors": row['authors'],
                           "img": row['small_image_url'],
                           "rating": math.ceil(row['average_rating'] * 2) / 2 ,
                    }
                    bookjson.append(data)

                testbook = {
                    "databooks": bookjson,
                    "face": "unknown"
                }
                
                # book_list = hybrid(int(2)).to_json
                yield "data: " + json.dumps(testbook) + "\n\n"
    return Response(live_stream(), mimetype= 'text/event-stream')

@app.route('/s')
def Home():
    return "<h1>Hello World!</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)