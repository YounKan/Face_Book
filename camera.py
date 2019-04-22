import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import numpy as np
import pickle
from flask import Flask, render_template

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
        self.count = 0
        self.facerec = ""
        with open("facerec.clf", 'rb') as f:
            self.knn_clf = pickle.load(f)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):

        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

        if success:
            # self.count +=1
            # cv2.imwrite("frame%d.jpg" % self.count, image)  
            # print("- Found {}".format(self.count))
            
            # if process_this_frame:
            # # Resize frame of video to 1/4 size for faster face recognition processing
            # small_frame = cv2.resize(image, (0, 0), fx=1, fy=1)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            # rgb_small_frame = image[:, :, ::-1]

            # Load image file and find face locations
            # X_img = face_recognition.load_image_file(image)
            X_face_locations = face_recognition.face_locations(image)

            if len(X_face_locations) == 0:
                self.count = 1 
                ret, jpeg = cv2.imencode('.jpg', image)
                return jpeg.tobytes()

            # # Find encodings for faces in the test iamge
            faces_encodings = face_recognition.face_encodings(image, known_face_locations=X_face_locations)

            # print("- Found {}".format(self.knn_clf.predict(faces_encodings)))

            # Use the KNN model to find the best matches for the test face
            closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= 0.4 for i in range(len(X_face_locations))]

            predictions =  [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(self.knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

            for name, (top, right, bottom, left) in predictions:
                
                if self.facerec == name:
                    self.count += 1
                else:
                    self.facerec = name
                    self.count = 1 
                print("name {} - count {}".format(self.facerec,self.count))
                # print("- Found {} at ({}, {})".format(name, left, top))
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top -= 20
                # right *= 4
                bottom += 20
                # left *= 4

                # Draw a box around the face
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()