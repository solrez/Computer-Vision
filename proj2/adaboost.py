
from utils import *
from weakclassifier import weakclassif
from matplotlib     import pyplot as plt
from haarFeature    import Feature
from sklearn.metrics import roc_curve
import numpy as np
import pdb

def trained_model(inp1 = None, label = None, filename = "", lim = 0):
  
    f1open = open(filename, "r+")
    file_len1 = f1open.readlines()

    classw1Num = int(len(file_len1) / 4)
    model     = AdaBoost(train = False, lim = classw1Num)

    if lim < classw1Num:
        model.weakclasslim = lim
    else:
        model.weakclasslim = classw1Num

    for i in range(0, len(file_len1), 4):

        alpha, dimension, direct1, threshold = None, None, None, None

        for j in range(i, i + 4):
            if   (j % 4) == 0:
                alpha     = float(file_len1[j])
            elif (j % 4) == 1:
                dimension = int(file_len1[j])
            elif (j % 4) == 2:
                direct1 = float(file_len1[j])
            elif (j % 4) == 3:
                threshold = float(file_len1[j])

        classif = model.classw1(train = False)
        classif.constructor(dimension, direct1, threshold)
        classif.img1 = inp1
        classif.label = label

        if inp1 is not None:
            classif.num_samples = inp1.shape[1]

        model.H[int(i/4)]     = classif
        model.alpha[int(i/4)] = alpha
        model.N         += 1

    model.img1 = inp1
    model.label = label
    if model.N > lim:
        model.N    = lim

    if label is not None:
        model.n_samples = len(label)

    f1open.close()

    return model


class AdaBoost:
    def __init__(self, inp1 = None, lab2 = None, classif = weakclassif, train = True, lim = 4):
        if train == True:
            self.img1   = inp1
            self.label = lab2

            self.sample_shape, self.n_samples = self.img1.shape

            
            assert self.n_samples == self.label.size

            self.num_face = np.count_nonzero(self.label == face_label)
            self.num_nonface = np.count_nonzero(self.label == nonface_label)

            
            face_w = [1.0/(2 * self.num_face) for i in range(self.num_face)]

            nonface_w = [1.0/(2 * self.num_nonface) for i in range(self.num_nonface)]
            self.W = np.array(face_w + nonface_w)

            self.acc1 = []

        self.classw1 = classif

        self.weakclasslim = lim

        self.H      = [None for _ in range(lim)]
        self.alpha  = [  0  for _ in range(lim)]
        self.N      = 0
        self.det_rate = 0.

        # true positive rate
        self.tpr = 0.
        self.fpr = 0.
        self.thresh  = 0.


    def stop_check(self):

        output = self.prediction(self.img1, self.thresh)
        right_dec = np.count_nonzero(output == self.label)/(self.n_samples*1.)
        self.acc1.append( right_dec)
        self.det_rate = np.count_nonzero(output[0:self.num_face] == face_label) * 1./ self.num_face
        ntp,nfn,ntn,nfp = 0 ,0,0,0
        for i in range(self.n_samples):
            if self.label[i] == face_label:
                if output[i] == face_label:
                    ntp += 1
                else:
                    nfn += 1
            else:
                if output[i] == face_label:
                    nfp += 1
                else:
                    ntn += 1
        self.tpr = ntp * 1./(ntp + nfn)
        self.fpr = nfp * 1./(ntn + nfp)

        if self.tpr > tpr_final and self.fpr < fpr_final:
            return True

    def train(self):

        for m in range(self.weakclasslim):
            self.N += 1

            #pdb.set_trace()
            self.H[m] = self.classw1(self.img1, self.label, self.W)
            #pdb.set_trace()
            err_rate = self.H[m].train()

            # if err_rate < 0.0001:
            #     err_rate = 0.0001

            epsilon1 = err_rate / (1 - err_rate)
            self.alpha[m] = np.log(1/epsilon1)
            output = self.H[m].prediction(self.img1)
            for i in range(self.n_samples):
                if self.label[i] == output[i]:
                    self.W[i] *=  epsilon1

            self.W /= sum(self.W)
            if fl_B is True:
                self.thresh, self.det_rate = self.optimal_thresh(tpr_final)

            if self.stop_check():
                print ("Training Done!")
                break

            if fl_T is True:
                print("weakclassif:", self.N)
                print("Error rate     :", err_rate)
                print("Accuracy      :", self.acc1[-1])
                print("Detection rate :", self.det_rate)
                print("Threshold :", self.thresh)
                print("alpha :", self.alpha[m])

        self.accuracy_cal()
        self.makeimg_classif()

        output = self.prediction(self.img1, self.thresh)
        lab=self.label
        dd=self.W
        #pdb.set_trace()
        plt.figure()
        fpr1, tpr1, _ = roc_curve(lab,output)#np.multiply(dd,output)
        plt.plot(fpr1, tpr1, color='darkorange')
        plt.savefig("ROC.jpg")
        return output, self.fpr

    def prediction(self, inp1, thresh = None):
        if thresh==None:
            thresh=self.thresh
        num_samples = inp1.shape[1]
        output = np.zeros(num_samples, dtype = np.float16)
        for i in range(self.N):
            output += self.H[i].prediction(inp1) * self.alpha[i]
    
        for i in range(len(output)):
            if output[i] > thresh:
                output[i] = face_label
            else:
                output[i] = nonface_label

        return output


    def optimal_thresh(self, tpr_final):
        det_rate = 0.
        best_thresh = None
        lb = -sum(self.alpha)
        ub = +sum(self.alpha)
        st      = -0.1
        threshold = np.arange(ub - st, lb + st, st)

        for t in range(threshold.size):

            output = self.prediction(self.img1, threshold[t])

            ntp = 0 
            nfn = 0 
            ntn = 0 
            nfp = 0 
            for i in range(self.n_samples):
                if self.label[i] == face_label:
                    if output[i] == face_label:
                        ntp += 1
                    else:
                        nfn += 1
                else:
                    if output[i] == face_label:
                        nfp += 1
                    else:
                        ntn += 1

            tpr = ntp * 1./(ntp + nfn)
            fpr = nfp * 1./(ntn + nfp)

            if tpr >= tpr_final:

                det_rate = np.count_nonzero(output[0:self.num_face] == face_label) * 1./ self.num_face

                best_thresh = threshold[t]
                break

        return best_thresh, det_rate

    def accuracy_cal(self):
        plt.figure()
        plt.title("Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy Prediction")
        plt.plot([i for i in range(self.N)], 
                    self.acc1, '-.', 
                    label = "Accuracy * 100%")
        plt.axis([0., self.N, 0, 1.])
        plt.savefig("Accuracy.jpg")
        



    def saveModel(self, filename):
        
        f1open = open(filename, "a+")

        for m in range(self.N):
            f1open.write(str(self.alpha[m]) + "\n")
            f1open.write(str(self.H[m].opti_shape) + "\n")
            f1open.write(str(self.H[m].opti_dir) + "\n")
            f1open.write(str(self.H[m].opti_threshold) + "\n")

        f1open.flush()
        f1open.close()

    def makeimg_classif(self):
        
        imgx  = img_x_size
        imgy = img_y_size

        haar = Feature(imgx, imgy)

        haarfeatures = haar.features
        best_features = [] 

        for n in range(self.N):
            best_features.append(haarfeatures[self.H[n].opti_shape])

        img_classif = np.zeros((imgy, imgx))

        for n in range(self.N):
            feature   = best_features[n]
            alpha     = self.alpha[n]
            direct1 = self.H[n].opti_shape

            (types, x, y, width, height) = feature

            image = np.array([[155 for i in range(imgx)] for j in range(imgy)])

            assert x >= 0 and x < imgx
            assert y >= 0 and y < imgy
            assert width > 0 and height > 0

            if direct1 == +1:
                black = bl
                white = wh
            else:
                black = wh
                white = bl

            if types == haar_type1:
                for i in range(y, y + height * 2):
                    for j in range(x, x + width):
                        if i < y + height:
                            image[i][j] = black
                        else:
                            image[i][j] = white

            elif types == haar_type2:
                for i in range(y, y + height):
                    for j in range(x, x + width * 2):
                        if j < x + width:
                            image[i][j] = white
                        else:
                            image[i][j] = black

            elif types == haar_type3:
                for i in range(y, y + height):
                    for j in range(x, x + width * 3):
                        if j >= (x + width) and j < (x + width * 2):
                            image[i][j] = black
                        else:
                            image[i][j] = white

            elif types == haar_type4:
                for i in range(y, y + height*3):
                    for j in range(x, x + width):
                        if i >= (y + height) and i < (y + height * 2):
                            image[i][j] = black
                        else:
                            image[i][j] = white

            else:
                for i in range(y, y + height * 2):
                    for j in range(x, x + width * 2):
                        if (j < x + width and i < y + height) or\
                           (j >= x + width and i >= y + height):
                            image[i][j] = white
                        else:
                            image[i][j] = black
            
            img_classif += image
            plt.figure()
            plt.matshow(image, cmap = "gray")
            plt.savefig(fig_dir + "features" + str(n) + ".jpg")
            
        from image import Image_perp
        plt.figure()
        img_classif = Image_perp.Normimg(img_classif)
        plt.matshow(img_classif, cmap = "gray")
        plt.savefig(fig_dir + "boosted_features.jpg")
