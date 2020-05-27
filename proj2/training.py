
from utils import *
from haarFeature import Feature
from image       import image_prep
from adaboost    import AdaBoost
from adaboost    import trained_model


import os
import numpy as np
import pdb

Face    = image_prep(face_dir,    num = face_samples)
nonFace = image_prep(nonface_dir, num = face_samples)

tot_samples = Face.num + nonFace.num
haar   = Feature(img_y_size, img_x_size)
if os.path.isfile(feature_dir + ".npy"):
    img1 = np.load(feature_dir + ".npy")
else:
   # pdb.set_trace()
    img1 = np.zeros((haar.featuresNum, tot_samples))
    #pdb.set_trace()
    for i in range(Face.num):
        featureVec = haar.calFeatureForImg(Face.images[i])
        for j in range(haar.featuresNum):
            img1[j][i]  = featureVec[j]
    for i in range(nonFace.num):
        featureVec = haar.calFeatureForImg(nonFace.images[i])
        for j in range(haar.featuresNum):
            img1[j][i + Face.num] = featureVec[j]

        np.save(feature_dir, img1)
mat = img1

featureNum, num = img1.shape
#pdb.set_trace()
assert num  == (face_samples + nonface_samples)
assert featureNum == feature_num

Label_Face    = [+1 for i in range(face_samples)]
Label_NonFace = [-1 for i in range(nonface_samples)]

label = np.array(Label_Face + Label_NonFace)

cache_filename = model_dir 

model = AdaBoost(mat, label, lim = num_weak_class)
model.train()
model.saveModel(cache_filename)

print(model)
