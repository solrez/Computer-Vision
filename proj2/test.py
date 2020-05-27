
from utils import *
from haarFeature import Feature
from image       import image_prep
from adaboost    import AdaBoost
from adaboost    import trained_model
import os
import numpy as np
import pdb

Face    = image_prep(face_dir_test,    num = face_samples_test)
nonFace = image_prep(nonface_dir_test, num = nonface_samples_test)

tot_samples = Face.num + nonFace.num
haar   = Feature(img_y_size, img_x_size)
if os.path.isfile(feature_dir_test + ".npy"):
    img1 = np.load(feature_dir_test + ".npy")
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

        np.save(feature_dir_test, img1)
mat = img1

featureNum, num = img1.shape
#pdb.set_trace()
assert num  == (face_samples_test + nonface_samples_test)
assert featureNum == feature_num

Label_Face    = [+1 for i in range(face_samples_test)]
Label_NonFace = [-1 for i in range(nonface_samples_test)]

label = np.array(Label_Face + Label_NonFace)

cache_filename = model_dir 

if os.path.isfile(cache_filename):

    model = trained_model(inp1     = img1,
                              label   = label,
                              filename= cache_filename,
                              lim   = num_weak_class)
    t=model.prediction(img1)
    M=0
    for i in range(len(t)):
        if t[i]!=label[i]:
            M+=1
    print('Error:',M/len(t))
    

