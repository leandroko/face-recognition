# USAGE
# python recognize_faces_video_multi_cam.py --encodings encodings.pickle --detection-method hog

# import the necessary packages
from imutils.video import VideoStream
import imutils
import face_recognition
import argparse
import pickle
import time
import cv2
import datetime
import numpy as np
import pandas as pd
import os
import threading

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

path = os.getcwd() + r'\registros'
registros = {}

for name in [name for name in os.listdir(os.getcwd() + '\\dataset')]:
    registros[name]={}
    registros[name]['datareg'] = ""
    registros[name]['entrada'] = ""
    registros[name]['saida'] = ""
    registros[name]['contador'] = 0

class camThread(threading.Thread):
    def __init__(self, previewName, camID, v):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.v = v
    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID, self.v)

# do a bit of cleanup
#cv2.destroyAllWindows()
#vs.stop()

# check to see if the video writer point needs to be released
#if writer is not None:
#	writer.release()
 
def camPreview(previewName, camID, v):
    cv2.namedWindow(previewName)
    #camera
    #cap = cv2.VideoCapture(camID)
    #video
    #cap = cv2.VideoCapture('v5.mp4')
    cap = VideoStream(src=camID).start()
    writer = None
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        frame = cap.read()
        
        # convert the input frame from BGR to RGB then resize it to have
        # a width of 750px (to speedup processing)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])
        
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb,
			model=args["detection_method"])
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        
        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
				encoding, tolerance=0.6)
            
            name = "Unknown"
            
            
            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                
                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)
            
            # código de captura dos quadros, nesse exemplo, para popular a base por minuto no lugar do dia
            if datetime.datetime.now().minute > 30:
                dia = datetime.datetime.now().minute - 30
            else:
                dia = datetime.datetime.now().minute
            
            # código para trazer dia ao invés do modelo de testes por minuto
            # dia = datetime.datetime.now().day
            
            mes = datetime.datetime.now().month
            ano = datetime.datetime.now().year
            datareg = str(dia) + '.' + str(mes) + '.' + str(ano)
            
            if registros[name]['datareg'] != datareg:
                registros[name]['entrada'] = ""
            
            registros[name]['datareg'] = datareg
            
            
            #if registros[name]['datareg'] != datareg:
            #	registros[name]['entrada'] == ""
            
            if registros[name]['entrada'] == "":
                registros[name]['entrada'] = datetime.datetime.now()
                
            registros[name]['saida'] = datetime.datetime.now()
            registros[name]['contador']+=1
            
            pd.DataFrame(registros).to_csv(path + r'\registros - ' + datareg + '.csv')
            print(registros)
            #print(name, datetime.datetime.now())
            
            # update the list of names
            names.append(name)
            
            
            if name == "Unknown":
                cv2.imwrite('unknown/'+str(datetime.datetime.now().strftime("%Y-%m-%d %H.%M"))+'.jpg', frame)
            
        # loop over the recognized faces
        
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)
            
            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
				(0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
				0.75, (0, 255, 0), 2)
            
        # if the video writer is None *AND* we are supposed to write
        # the output video to disk initialize the writer
        if writer is None and args["output"] is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 20,
				(frame.shape[1], frame.shape[0]), True)
            
        # if the writer is not None, write the frame with recognized
        # faces t odisk
        if writer is not None:
            writer.write(frame)
            
        # check to see if we are supposed to display the output frame to
        # the screen
        if args["display"] > 0:
            cv2.imshow(previewName, frame)
            key = cv2.waitKey(1) & 0xFF
            
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    cap.release()

# Create threads as follows
thread1 = camThread("Camera 1", 0, 0)
thread2 = camThread("Camera 2", 1, 0)

thread1.start()
thread2.start()

print()
print("Active threads", threading.activeCount())