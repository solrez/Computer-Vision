import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import image

class Image_perp:

    def __init__(self, fileName = None, label = None, input1 = None):
        # imort
        if fileName != None:
            self.imgName = fileName
            self.img     = image.imread(fileName)

            if len(self.img.shape) == 3:
                self.img     = self.img[:,:, 1]
        else:
            self.img     = input1

        self.label   = label


        self.vec = Image_perp.Integralimg( Image_perp.Normimg(self.img)  ).transpose().flatten()

    def Integralimg(image):
        row, col = image.shape
        img = np.zeros((row, col))
        img = image.cumsum(axis=1).cumsum(axis=0)
        return img

    
    def Normimg(image):
        row, col = image.shape
        img_new = np.zeros((row, col))
        meanVal = image.mean()
        stdValue = image.std()
        if stdValue == 0:
            stdValue = 1

        img_new = (image - meanVal) / stdValue

        return img_new
    # def show(image = None):
    #     if image == None:
    #         return
    #     plt.input1show(image)
    #     plt.show()


class image_prep:
    def __init__(self, dir1 = None, label = None, num = None):

        assert isinstance(dir1, str)

        self.dir1 = dir1
        self.fileList = os.listdir(dir1)
        self.fileList.sort()

        if num == None:
            self.num = len(self.fileList)
        else:
            self.num = num

        self.label  = label

        self.images = [None for _ in range(self.num)]

        for i in range(self.num):
            self.images[i] = Image_perp(dir1 + self.fileList[i], label)

        print("The Data has been loaded\n")



