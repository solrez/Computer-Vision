

from matplotlib import pyplot as plt
import numpy as np

from utils import *

class weakclassif:

    def __init__(self, Mat = None, lab = None, W = None, train = True):


        if train == True:
 
            self.img = Mat
            self.label = lab
            self.shape_sample, self.num_sample = self.img.shape
            self.weight = W
            self.output = np.zeros(self.num_sample, dtype = np.int)
            self.opti_err = 1.
            self.opti_shape = 0
            self.opti_threshold = None
            self.opti_dir = 0
            

    def opti1(self, inp1):

        
        index1 = (self.label + face_label) / (face_label * 2)
        weight = self.weight  * index1
        vec = self.img[inp1] * index1
        face_s = weight.dot(vec)
        face_w= weight.sum()

        
        index1 = (self.label + nonface_label) / (nonface_label * 2)
        weight = self.weight  * index1
        vec = self.img[inp1] * index1
        nonface_s = weight.dot(vec)
        nonface_w= weight.sum()
                
        fac_cont = face_s / face_w
        nonface_cont = nonface_s / nonface_w

        threshold = (fac_cont + nonface_cont)/2

        minerr    = np.inf
        best_dir = None
        for dir1 in [-1, 1]:
            err = 0.

            self.output[self.img[inp1] * dir1 < threshold * dir1]\
                    = face_label

            self.output[self.img[inp1] * dir1 >= threshold * dir1]\
                    = nonface_label

            err = self.weight[ self.output != self.label].sum()

            self.output *= 0 # reset the output
            if err < minerr:
                minerr    = err
                best_dir = dir1

        return minerr, threshold, best_dir

    def train(self):

        for dim in range(self.shape_sample):
            err, threshold, dir1 = self.opti1(dim)
            if err < self.opti_err:
                self.opti_err = err
                self.opti_shape = dim
                self.opti_threshold = threshold
                self.opti_dir = dir1

        assert self.opti_err < 0.5
        self.disp_thresh(self.opti_shape)
        return self.opti_err
        

    def prediction(self, Mat):
        num_sample = Mat.shape[1]

        dim       = self.opti_shape
        threshold = self.opti_threshold
        dir1 = self.opti_dir

        output = np.zeros(num_sample, dtype = np.int)

        output[Mat[dim] * dir1 <  dir1 * threshold] = face_label
        output[Mat[dim] * dir1 >= dir1 * threshold] = nonface_label

        return output

    def disp_thresh(self, dim = None):

        N = 10 # the number of center
        maxn = np.max(self.img[dim])
        minn = np.min(self.img[dim])

        scope = (maxn - minn) / N

        cens = [ (minn - scope/2)+ scope*i for i in range(N)]
        count = [ [0, 0] for i in range(N)]

        for j in range(N):
            for i in range(self.num_sample):
                if abs(self.img[dim][i] - cens[j]) < scope/2:
                    if self.label[i] == face_label:
                        count[j][1] += 1
                    else:
                        count[j][0] += 1

        facev, nonfacev = [], []

        for i in range(N):
            facev.append(count[i][1])
            nonfacev.append(count[i][0])

        face_sVal = sum(facev)
        nonface_sVal = sum(nonfacev)

        for i in range(len(facev)): facev[i] /= (1. * face_sVal)
        for i in range(len(nonfacev)): nonfacev[i] /= (1. * nonface_sVal)

        plt.title(" weak classifier with threshold")
        plt.plot(cens, facev, "r-o", label = "Face class")
        plt.plot(cens, nonfacev, "b-o", label = "Non-Face class")
        plt.xlabel("feature")
        plt.ylabel("frequency")
        

        # plot threshold line
        face_w = 0.
        nonface_w = 0.
        face_s = 0.
        nonface_s = 0.
        for i in range(self.num_sample):
            if self.label[i] == face_label:
                face_s  += self.weight[i] * self.img[dim][i]
                face_w += self.weight[i]
            else:
                nonface_s  += self.weight[i] * self.img[dim][i]
                nonface_w += self.weight[i]
                
        fac_cont = face_s / face_w
        nonface_cont = nonface_s / nonface_w

        threshold = (fac_cont + nonface_cont)/2
        plt.plot([threshold for i in range(10)], [i for i in np.arange(0.0, 0.5, 0.05)], label = "threshold")
        #plt.legend()
        plt.savefig(fig_dir + "simple_classfication11.jpg")
        #plt.disp_thresh()

    def __str__(self):

        string  = "opti_err:" + str(self.opti_err) + "\n"
        string += "opti_threshold:" + str(self.opti_threshold) + "\n"
        string += "opti_shape:" + str(self.opti_shape) + "\n"
        string += "opti_dir:" + str(self.opti_dir) + "\n"
        string += "weights      :" + str(self.weight)        + "\n"
        return string

    def constructor(self, dimension, dir1, threshold):
        self.opti_shape = dimension
        self.opti_threshold = threshold
        self.opti_dir = dir1

        return self
