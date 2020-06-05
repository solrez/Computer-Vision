import numpy as np
from utils import *
from image import Image_perp
from matplotlib import pyplot as plt


class Feature:
    def __init__(self, img_Width, img_Height):

        self.featureName = "Haar Feature"

        self.img_Width  = img_Width
        self.img_Height = img_Height

        self.tot_pixels = img_Width * img_Height

        self.featureTypes = (haar_type1,
                             haar_type2,
                             haar_type3,
                             haar_type4,
                             haar_type5)

        self.features    = self._evalFeatures_total()

        self.featuresNum = len(self.features)
        self.vector = np.zeros(self.featuresNum, dtype=np.float32)
        self.idxVector_tmp_0 = np.zeros(self.tot_pixels, dtype = np.int8)
        self.idxVector_tmp_1 = np.zeros(self.tot_pixels, dtype = np.int8)
        self.idxVector_tmp_2 = np.zeros(self.tot_pixels, dtype = np.int8)
        self.idxVector_tmp_3 = np.zeros(self.tot_pixels, dtype = np.int8)


    def vecRectSum(self, idxVector, x, y, width, height):
        idxVector *= 0 # reset this vector
        if x == 0 and y == 0:
            idxVector[width * height + 2] = +1
        elif x == 0:
            idx1 = self.img_Height * (    width - 1) + height + y - 1
            idx2 = self.img_Height * (    width - 1) +          y - 1
            idxVector[idx1] = +1
            idxVector[idx2] = -1

        elif y == 0:
            idx1 = self.img_Height * (x + width - 1) + height - 1
            idx2 = self.img_Height * (x         - 1) + height - 1
            idxVector[idx1] = +1
            idxVector[idx2] = -1
        else:
            idx1 = self.img_Height * (x + width - 1) + height + y - 1
            idx2 = self.img_Height * (x + width - 1) +          y - 1
            idx3 = self.img_Height * (x         - 1) + height + y - 1
            idx4 = self.img_Height * (x         - 1) +          y - 1

            assert idx1 < self.tot_pixels and idx2 < self.tot_pixels 
            assert idx3 < self.tot_pixels and idx4 < self.tot_pixels 

            idxVector[idx1] = + 1
            idxVector[idx2] = - 1
            idxVector[idx3] = - 1
            idxVector[idx4] = + 1

        return idxVector


    def vec_haartypeI(self, vec, x, y, width, height):
        vec1 = self.vecRectSum(self.idxVector_tmp_0, x, y         , width, height)
        vec2 = self.vecRectSum(self.idxVector_tmp_1, x, y + height, width, height)

        featureSize = width * height * 2

        return (vec1.dot(vec) - vec2.dot(vec))/featureSize


    def vec_haartypeII(self, vec, x, y, width, height):
        vec1 = self.vecRectSum(self.idxVector_tmp_0, x + width, y, width, height)
        vec2 = self.vecRectSum(self.idxVector_tmp_1, x        , y, width, height)

        featureSize = width * height * 2

        return (vec1.dot(vec) - vec2.dot(vec))/featureSize


    def vec_haartypeIII(self,vec, x, y, width, height):
        vec1 = self.vecRectSum(self.idxVector_tmp_0, x +   width, y, width, height)
        vec2 = self.vecRectSum(self.idxVector_tmp_1, x          , y, width, height)
        vec3 = self.vecRectSum(self.idxVector_tmp_2, x + 2*width, y, width, height)

        featureSize = width * height * 3

        return (vec1.dot(vec) - vec2.dot(vec)
                - vec3.dot(vec))/featureSize


    def vec_haartypeIV(self,vec, x, y, width, height):
        vec1 = self.vecRectSum(self.idxVector_tmp_0, x, y +   height, width, height)
        vec2 = self.vecRectSum(self.idxVector_tmp_1, x, y           , width, height)
        vec3 = self.vecRectSum(self.idxVector_tmp_2, x, y + 2*height, width, height)

        featureSize = width * height * 3

        return (vec1.dot(vec) - vec2.dot(vec)
                - vec3.dot(vec))/featureSize


    def vec_haartypeV(self, vec, x, y, width, height):
        vec1 = self.vecRectSum(self.idxVector_tmp_0, x + width,          y, width, height)
        vec2 = self.vecRectSum(self.idxVector_tmp_1, x        ,          y, width, height)
        vec3 = self.vecRectSum(self.idxVector_tmp_2, x        , y + height, width, height)
        vec4 = self.vecRectSum(self.idxVector_tmp_3, x + width, y + height, width, height)

        featureSize = width * height * 4

        return (vec1.dot(vec) - vec2.dot(vec) +
                vec3.dot(vec) - vec4.dot(vec))/featureSize


    def _evalFeatures_total(self):
        win_Height = self.img_Height
        win_Width  = self.img_Width

        height_Limit = {haar_type1    : win_Height/2 - 1,
                         haar_type2  : win_Height   - 1,
                         haar_type3 : win_Height   - 1,
                         haar_type4  : win_Height/3 - 1,
                         haar_type5   : win_Height/2 - 1}

        width_Limit  = {haar_type1   : win_Width   - 1,
                        haar_type2  : win_Width/2 - 1,
                        haar_type3 : win_Width/3 - 1,
                        haar_type4  : win_Width   - 1,
                        haar_type5   : win_Width/2 - 1}

        features = []
        for types in self.featureTypes:
            for w in range(1, int(width_Limit[types])):
                for h in range(1,int(height_Limit[types])):

                    if w == 1 and h == 1:
                        continue

                    if types == haar_type1:

                        x_limit = win_Width  - w
                        y_limit = win_Height - 2*h
                        for x in range(1, x_limit):
                            for y in range(1, y_limit):
                                features.append( (types, x, y, w, h))

                    elif types == haar_type2:
                        x_limit = win_Width  - 2*w
                        y_limit = win_Height - h
                        for x in range(1, x_limit):
                            for y in range(1, y_limit):
                                features.append( (types, x, y, w, h))

                    elif types == haar_type3:
                        x_limit = win_Width  - 3*w
                        y_limit = win_Height - h
                        for x in range(1, x_limit):
                            for y in range(1, y_limit):
                                features.append( (types, x, y, w, h))


                    elif types == haar_type4:
                        x_limit = win_Width  - w
                        y_limit = win_Height - 3*h
                        for x in range(1, x_limit):
                            for y in range(1, y_limit):
                                features.append( (types, x, y, w, h))

                    elif types == haar_type5:
                        x_limit = win_Width  - 2*w
                        y_limit = win_Height - 2*h
                        for x in range(1, x_limit):
                            for y in range(1, y_limit):
                                features.append( (types, x, y, w, h))

        return features


    def calFeatureForImg(self, img):

        assert isinstance(img, Image_perp)
        assert img.img.shape[0] == self.img_Height
        assert img.img.shape[1] == self.img_Width

        for i in range(self.featuresNum):
            type, x, y, w, h = self.features[i]

            if   type == haar_type1:
                self.vector[i] = self.vec_haartypeI(img.vec, x, y, w, h)
            elif type == haar_type2:
                self.vector[i] = self.vec_haartypeII(img.vec, x, y, w, h)
            elif type == haar_type3:
                self.vector[i] = self.vec_haartypeIII(img.vec, x, y, w, h)
            elif type == haar_type4:
                self.vector[i] = self.vec_haartypeIV(img.vec, x, y, w, h)
            elif type == haar_type5:
                self.vector[i] = self.vec_haartypeV(img.vec, x, y, w, h)
            else:
                raise Exception("unknown feature type")

        return self.vector

    def makeFeaturePic(self, feature):

        (types, x, y, width, height) = feature

        assert x >= 0 and x < self.img_Width
        assert y >= 0 and y < self.img_Height
        assert width > 0 and height > 0

        image = np.array([[125. for i in range(self.img_Width)]
                                 for j in range(self.img_Height)])

        if types == haar_type1:
            for i in range(y, y + height * 2):
                for j in range(x, x + width):
                    if i < y + height:
                        image[i][j] = bl
                    else:
                        image[i][j] = wh

        elif types == haar_type2:
            for i in range(y, y + height):
                for j in range(x, x + width * 2):
                    if j < x + width:
                        image[i][j] = wh
                    else:
                        image[i][j] = bl

        elif types == haar_type3:
            for i in range(y, y + height):
                for j in range(x, x + width * 3):
                    if j >= (x + width) and j < (x + width * 2):
                        image[i][j] = bl
                    else:
                        image[i][j] = wh

        elif types == haar_type4:
            for i in range(y, y + height*3):
                for j in range(x, x + width):
                    if i >= (y + height) and i < (y + height * 2):
                        image[i][j] = bl
                    else:
                        image[i][j] = wh

        elif types == haar_type5:
            for i in range(y, y + height * 2):
                for j in range(x, x + width * 2):
                    if (j < x + width and i < y + height) or\
                       (j >= x + width and i >= y + height):
                        image[i][j] = bl
                    else:
                        image[i][j] = wh


        plt.matshow(image, cmap = "gray")
        plt.show()



