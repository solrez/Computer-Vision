
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
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

def mean_cov(a):
    Mean = np.mean(a, axis=0)
    Cov = np.cov(np.transpose(a))
    return Mean, Cov

def Gauss_dist(a, Cov, Mean):
    log_pdf = np.zeros(len(a))
    for i in range(len(a)):
        d = np.matmul((a[i] - Mean),(inv(Cov)))
        d1 = np.matmul(d,((a[i] - Mean).transpose()))
        pdf = np.exp(-0.5*d1)
        log_pdf[i] = np.log(pdf)
    return log_pdf

def Posterior(a, b):
    post_face = a/(a + b)
    Post_non = b/(a + b)
    return post_face, Post_non

img_train_face = Data_loader_img(1)
img_train_non = Data_loader_img(2)
img_test_face = Data_loader_img(3)
img_test_non = Data_loader_img(4)

mean_face, cov_face = mean_cov(img_train_face)
mean_non, cov_non = mean_cov(img_train_non)

mean_esti_face=np.reshape(mean_face,(10,10))
mean_esti_non=np.reshape(mean_non,(10,10))

cv2.imwrite('q1_mean_face.jpg', mean_esti_face)
cv2.imshow('ii',mean_esti_face)
cv2.imwrite('q1_mean_non.jpg', mean_esti_non)
cv2.imshow('i',mean_esti_non)
cv2.imwrite('q1_covar_face.jpg', cov_face)
cv2.imshow('l',cov_face)
cv2.imwrite('q1_covar_nonface.jpg', cov_non)
cv2.imshow('k',cov_non)

p_ff = Gauss_dist(img_test_face, cov_face, mean_face)
p_nf = Gauss_dist(img_test_non, cov_face, mean_face)

p_fn = Gauss_dist(img_test_face, cov_non, mean_non)
p_nn = Gauss_dist(img_test_non, cov_non, mean_non)

post_ff, post_fn  = Posterior(p_ff, p_fn)
post_nf, post_nn  = Posterior(p_nf, p_nn)

false_positive = 0
for i in range(len(post_ff)):
    if post_nf[i]> 0.5:
        false_positive=false_positive+1
print ("False Positive Rate ", float(false_positive)/len(post_ff))

false_negative = 0
for i in range(len(post_ff)):
    if post_fn[i]> 0.5:
        false_negative=false_negative+1
print ("False Negative Rate ", float(false_negative)/len(post_ff))

misclassification = (float(false_positive + false_negative))/ (len(post_ff) + len(post_ff))
print ("Misclassification Rate", misclassification)

Posterior = np.append( post_ff, post_nf)
labels = np.append(np.ones(100), np.zeros(100)   )

fpr, tpr, _ = roc_curve(labels, Posterior, pos_label=0)
plt.plot(fpr, tpr, color='darkorange')
plt.xlabel("false positive rate")
plt.ylabel("true positive rate" )
plt.show()
