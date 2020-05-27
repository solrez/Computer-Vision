import cv2
import os
from random import *
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import digamma
from scipy.special import gamma
from numpy.linalg import inv
import scipy
from scipy import optimize
from sklearn.metrics import roc_curve

def Data_loader_img(n):
    if n == 1:
        image_dir = "rsoleim_project01/data/face/train/"
    elif n==2:
        image_dir = "rsoleim_project01/data/non_face/train/"
    elif n==3:
        image_dir = "rsoleim_project01/data/face/test/"
    else:
        image_dir = "rsoleim_project01/data/non_face/test"

    image_vec = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            image_vec.append(os.path.join(root, file))
    out_image = []
    for imageData_loader_img in image_vec:
        img = cv2.imread(imageData_loader_img)
        img = cv2.resize(img, (10,10))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_prep = img_gray.flatten()
        out_image.append(img_prep)
    return out_image

def mean_conv_D_nu():

    mean = np.random.randint(0,255, size=(1, 100))
    covar1=np.zeros((100,100))
    for j in range(100):
        for k in range(100):
            if j==k:
                covar1[j][k]=int((np.random.random()))
            else:
                covar1[j][k]=0
    covar1 = covar1 + 1*np.eye(100)
    nu,D =10,10 
    return  mean, covar1, nu, D


def E_step( img_train_face, mean, covar1, nu, D):
    
    mean_h = np.zeros(len(img_train_face))
    mean_hi = np.zeros(len(img_train_face))
    x = np.zeros(len(img_train_face))

    for i in range(len(img_train_face)):
        x[i] = np.matmul(np.matmul((img_train_face[i] - mean),(inv(covar1[:,:]))),((img_train_face[i] - mean).transpose()))
        mean_h[i]=(nu+D)/(nu+x[i])
        mean_hi[i] = digamma(((nu/2) + (D/2)), out=None) - np.log((nu/2) + (x[i]/2))

    return mean_h, mean_hi, x

def M_step(mean_h, mean_hi,  img_train_face, nu ):
    x = np.sum(mean_h)
    ax_vec = np.asarray(img_train_face)
    h_ti = np.zeros((len(img_train_face),100))
    for i in range(len(img_train_face)):
        h_ti[i] = mean_h[i]*ax_vec[i]
    x_ti = np.sum(h_ti, axis=0)
    mean_update = x_ti/x

    covar1_ti = np.zeros((100,100, len(img_train_face)))
    for i in range(len(img_train_face)):
       covar1_ti[:,:,i] = np.matmul(((ax_vec[i]-mean).transpose()), mean_h[i]*(ax_vec[i]-mean))
    covar1_update = (np.sum(covar1_ti, axis=2))/x
    covar1_update = np.diag(np.diag(covar1_update))

    def T_dist(nu):
        prob = (len(img_train_face) * ((nu/2)* np.log(nu/2))) + (len(img_train_face)* np.log(gamma(nu/2))) - (((nu/2)-1)* np.sum(mean_hi)) + ((nu/2)*np.sum(mean_h))
        return prob

    nu_opt = scipy.optimize.fmin(T_dist, nu)
    nu_update = nu_opt[0]

    return mean_update, covar1_update, nu_update

def prob_dist(img_train_face, mean, covar1, nu, D):
    y = np.zeros(len(img_train_face))
    p_out = np.zeros(len(img_train_face))
    for i in range(len(img_train_face)):
        y[i] = np.matmul( np.matmul((img_train_face[i] - mean),(inv(covar1[:,:]))),((img_train_face[i] - mean).transpose()))
        p_out[i] = (gamma((nu+D)/2) * ((1 + (y[i]/nu))**(-(nu+D)/2)))/gamma(nu/2)
    return p_out

def Posterior(a, b):
    post_face = a/(a + b)
    Post_non = b/(a + b)
    return post_face, Post_non

num_it = 7
img_train_face = Data_loader_img(1)
img_train_non = Data_loader_img(2)
img_test_face = Data_loader_img(3)
img_test_non = Data_loader_img(4)

mean, covar1, v, D = mean_conv_D_nu()
mean_non, covar1_non, v_non, D_non = mean_conv_D_nu()

for it in range(num_it):
    if it < num_it:
        print ("it: ", it)
        mean_h, mean_hi, d1 = E_step(img_train_face, mean, covar1, v, D)
        mean, covar1, v = M_step(mean_h, mean_hi,  img_train_face, v )
        mean_h_non, mean_hi_non, d1_non = E_step(img_train_non, mean_non, covar1_non, v_non, D_non)
        mean_non, covar1_non, v_non = M_step(mean_h_non, mean_hi_non,  img_train_non, v_non )

mean_esti_face=np.reshape(mean,(10,10))
mean_esti_non=np.reshape(mean_non,(10,10))
cv2.imwrite('mean_face_t-distribution.jpg', mean_esti_face)
cv2.imshow('ii',mean_esti_face)
cv2.imwrite('mean_nonface_t-distribution.jpg', mean_esti_non)
cv2.imshow('i',mean_esti_non)
cv2.imwrite('covar1_face_tdist_new'+'.jpg',covar1)
cv2.imshow('lt',covar1)
cv2.imwrite('covar1_nonface_tdist_new'+'.jpg',covar1_non)
cv2.imshow('l',covar1_non)

p_ff=  prob_dist(img_test_face, mean, covar1, v, D)
p_fn =  prob_dist(img_test_face, mean_non, covar1_non, v_non, D_non)
p_nf =  prob_dist(img_test_non, mean, covar1, v, D)
p_nn=  prob_dist(img_test_non, mean_non, covar1_non, v_non, D_non)

post_ff, post_fn  = Posterior(p_ff, p_fn)
post_nf, post_nn  = Posterior(p_nf, p_nn)

false_positive = 0
for i in range(len(post_ff)):
    if post_nf[i]> 0.5:
        false_positive=false_positive+1
print ("False Positive Rate ", float(false_positive)/100)

false_negative = 0
for i in range(len(post_ff)):
    if post_fn[i]> 0.5:
        false_negative = false_negative + 1
print ("False Negative Rate ", float(false_negative)/100)

misclassification = (float(false_positive + false_negative)) / (len(post_ff) + len(post_ff))
print ("Misclassification Rate", misclassification)

Posterior = np.append(post_ff, (post_nf))
labels = np.append(np.ones(100), np.zeros(100))

fpr, tpr, _ = roc_curve(labels, Posterior, pos_label=1)
plt.plot(fpr, tpr, color='darkorange')
plt.xlabel("false positive rate")
plt.ylabel("true positive rate" )
plt.show()
