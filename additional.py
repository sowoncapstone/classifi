#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:21:46 2019

@author: dongwon.j
"""

import pickle
import os
import numpy as np
import argparse

##
who = "전동원"
userID = "dongwon0207"
##

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--who", required=True, help="who")
ap.add_argument("-i", "--id", required=True, help="ID")
args = vars(ap.parse_args())

who = args["who"]
userID = args["id"]

waitData = np.array(pickle.loads(open("waiting/{}/waiting.pickle".format(userID), "rb").read()))

dataDir = "data/{}".format(userID)
if not os.path.isdir(dataDir):
    os.mkdir(dataDir)
    
    criteria = []
    criteria.append({"name":who, "points":np.array([waitData]), "avg":waitData, "std":np.array([0 for a in range(0, len(waitData))]), "num":1})
    
    with open("data/{}/criteria.pickle".format(userID), "wb") as fw:
        pickle.dump(criteria, fw)
    os.remove("waiting/{}/waiting.pickle".format(userID))
    os.remove("image/{}/new.pickle".format(userID))
    os.remove("image/{}/png/{}.jpg".format(userID, userID))

else:
    fr = open("data/{}/criteria.pickle".format(userID), "rb")
    criteria = pickle.load(fr)
    check = 0
    
    for a in criteria:
        if a["name"] == who:
            check = 1
            a["num"] += 1;
            a["points"] = np.append(a["points"], np.array([waitData]), axis = 0)
            a["avg"] = np.average(a["points"], axis=0)
            bean = np.zeros([1, 128])[0]
            for b in a["points"]:
                bean += (b - a["avg"])**2
            a["std"] = np.sqrt(bean)
    
    if check == 0:
        criteria.append({"name":who, "points":np.array([waitData]), "avg":waitData, "std":np.array([0 for a in range(0, len(waitData))]), "num":1})
    
    with open("data/{}/criteria.pickle".format(userID), "wb") as fw:
        pickle.dump(criteria, fw)
    os.remove("waiting/{}/waiting.pickle".format(userID))
    os.remove("image/{}/new.pickle".format(userID))
    os.remove("image/{}/png/{}.jpg".format(userID, userID))
