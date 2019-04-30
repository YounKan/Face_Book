from flask import Flask, render_template, Response, url_for, request
from camera import VideoCamera
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD, evaluate
import json
import math
import pickle

app = Flask(__name__)

cache = {}

class DataStore():
    facerec = None
    tagrec = None

datalocal = DataStore()

databook = pd.read_json('data/booksdata.json')
tags = pd.read_json('data/bookt.json')
facedata = pd.read_csv('data/friend_face_id.csv')
ratings = pd.read_csv('data/ratings.csv')

with open('data/bookt.json') as json_file:  
    booktags = json.load(json_file)

# Content-based
tf =  TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(databook['feature'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

titles = databook[['title','authors','small_image_url','average_rating']]

# CF
with open('data/bookrec.clf', 'rb') as f:
    recbook = pickle.load(f)


def find_feature(title):
  feature =  databook[(databook['title'] == title)].index[0]
  return feature

def get_book(statusRec , Id):
  listbook = []
  if statusRec == "user":
    userbooks = ratings[(ratings['user_id'] == Id)]
    userbooks = userbooks.merge(databook[['title', 'book_id']], on='book_id')
    listbook = list(userbooks['title'])
  else:
    tagbooks = tags[(tags['tag_id'] == Id)]
    for book in tagbooks['books'].values[0]:
      listbook.append(book['title'])
      
  return listbook

def hybrid(statusRec , Id):
  
  titlebook = get_book(statusRec,Id)
  
  bookrecs = pd.DataFrame()

  for x in titlebook:
      idx = find_feature(x)
      sim_scores = list(enumerate(cosine_sim[idx]))
      sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
      sim_scores = sim_scores[1:31]
      book_indices = [i[0] for i in sim_scores]
      bookrec = databook.iloc[book_indices][['book_id','title','authors','small_image_url','average_rating','language_code']]
      bookrecs = pd.concat([pd.DataFrame(bookrec), bookrecs], ignore_index=True)
      bookrecs['est'] = bookrecs['book_id'].apply(lambda x: recbook.predict(Id, x).est)
      bookrecs = bookrecs.sort_values('est', ascending=False)
      
  return bookrecs.head(5)

def get_bookrec_api(book_list , face):

    bookjson = []
    for index, row in book_list.iterrows():
        data ={
            "title": row['title'],
            "authors": row['authors'],
            "img": row['small_image_url'],
            "rating": math.ceil(row['average_rating'] * 2) / 2 
        }
        bookjson.append(data)

    testbook = {
        "databooks": bookjson,
        "booktags": booktags,
        "face": face
    }
    return testbook

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if camera.count == 5:
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
                jsondata = {}
                if datalocal.facerec == "unknown":
                    jsondata = {
                        "databooks": "",
                        "booktags": booktags,
                        "face": "select"
                    }
                elif datalocal.facerec == "tag":
                    jsondata = get_bookrec_api(hybrid("tag",int(datalocal.tagrec)),"tag")
                else:
                    face = facedata[(facedata['user_id'] == datalocal.facerec)]
                    userid = face['face_id'].values[0]
                    jsondata = get_bookrec_api(hybrid("user",int(userid)),"know")
               
                datalocal.facerec = None
                yield "data: " + json.dumps(jsondata) + "\n\n"
    return Response(live_stream(), mimetype= 'text/event-stream')

@app.route('/select')
def save_data():
    datalocal.facerec = "tag"
    datalocal.tagrec = request.args['tag_id']
    return 'success'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)