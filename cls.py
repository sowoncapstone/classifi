from operator import itemgetter
from time import sleep
from imutils import paths
from PIL import Image

import numpy as np
import pickle
import os
import face_recognition
import argparse
import cv2
import base64


#####################################디코드 파트################

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path")
args = vars(ap.parse_args())

userID = args["dataset"].split("/")[-2]

f = open("image/{}/{}.txt".format(userID, userID), "r")
txtImage = f.read()

Dir = "image/{}/png".format(userID)
if not os.path.isdir(Dir):
	os.mkdir(Dir)

fh = open("image/{}/png/{}.png".format(userID, userID), "wb")
fh.write(base64.b64decode(txtImage))
fh.close()

im = Image.open("image/{}/png/{}.png".format(userID, userID))
im = im.convert("RGB")
im.save("image/{}/png/{}.jpg".format(userID, userID))
os.remove("image/{}/png/{}.png".format(userID, userID))


#####################################임베딩 파트################

imagePaths = list(paths.list_images(args["dataset"]))

knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
	name = imagePath.split(os.path.sep)[-2]

	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	boxes = face_recognition.face_locations(rgb, model="cnn")

	encodings = face_recognition.face_encodings(rgb, boxes)

	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

data = {"encodings": knownEncodings, "names": knownNames}
f = open("image/{}/new.pickle".format(userID), "wb")
f.write(pickle.dumps(data))
f.close()

######################################클래시피케이션 #############

Dir = "data/{}".format(userID)
if not os.path.isdir(Dir):
    data = pickle.loads(open("image/{}/new.pickle".format(userID), "rb").read())
    Xtra = np.array(data["encodings"])
    Xtra
    
    waitingDir = "waiting/{}".format(userID)
    
    if not os.path.isdir(waitingDir):
        os.mkdir(waitingDir)
        
    with open("waiting/{}/waiting.pickle".format(userID), "wb") as fw:
        pickle.dump(Xtra[0], fw)
    print("check")
    
else:
    fr = open("data/{}/criteria.pickle".format(userID), "rb")
    criteria = pickle.load(fr)
    
    newdata = pickle.loads(open("image/{}/new.pickle".format(userID), "rb").read())
    Xtes = np.array(newdata["encodings"])
    ytes = np.array(newdata["names"])
    
    result = []
    for a in criteria:
        result.append({"name": str(a["name"]), "value": np.sqrt(np.sum((Xtes-a["avg"])**2))})
    sortedresult = sorted(result, key = itemgetter("value"))
    who = sortedresult[0]["name"]
    value = sortedresult[0]["value"]
    
    for a in criteria:
        if a["name"] == who:
            new = np.mean((Xtes - a["avg"])/a["std"])
    
    if value >= 0.30:
        with open("waiting/{}/waiting.pickle".format(userID), "wb") as fw:
            pickle.dump(Xtes[0], fw)
        print("check")
        
    ##elif value >= 0.25 and value < 0.35:
    ##    print("check")
    
    else:
        for a in criteria:
            if a["name"] == who:
                a["num"] += 1;
                a["points"] = np.append(a["points"], np.array([Xtes[0]]), axis = 0)
                a["avg"] = np.average(a["points"], axis=0)
                bean = np.zeros([1, 128])[0]
                for b in a["points"]:
                    bean += (b - a["avg"])**2
                a["std"] = np.sqrt(bean)
                
        with open("data/{}/criteria.pickle".format(userID), "wb") as fw:
            pickle.dump(criteria, fw)
        os.remove("image/{}/new.pickle".format(userID))
        os.remove("image/{}/png/{}.jpg".format(userID, userID))
        print(who)
