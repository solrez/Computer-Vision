import cv2
import os
import random 
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

def mean_covar1_phi(img_train_face, k):

    mean = np.mean(img_train_face, axis=0)
    covar1 = np.diag(np.diag(np.cov(np.transpose(img_train_face))))
    phi = np.random.random_sample((100, k))

    return mean, covar1, phi

def E_step( k, img_train_face, phi, mean, covar1):

    mean_h = np.zeros((k, len(img_train_face)))
    mean_h_ti = np.zeros((k,k, len(img_train_face)))
    a = np.zeros((k,k,len( img_train_face)))

    for i in range(len(img_train_face)):
        ax_vec = np.matmul(np.matmul(phi.transpose(),inv(covar1)),phi) + np.eye(k)
        mean_h[:,i] = np.matmul(np.matmul(np.matmul(inv(ax_vec),phi.transpose()), inv(covar1)), ((img_train_face[i]- mean).transpose()))
        a[:,:,i] = mean_h[:,i]*(mean_h[:,i][np.newaxis]).transpose()
        mean_h_ti[:,:,i] = inv(ax_vec) + a[:,:,i]

    return mean_h, mean_h_ti

def M_step(k,mean_h, mean_h_ti,mean, img_train_face ):

    app_sum = np.sum(mean_h_ti, axis=2)

    y = np.zeros((100,k,len(img_train_face)))
    for i in range(len(img_train_face)):
        y[:,:,i] = np.matmul((img_train_face[i]-mean)[np.newaxis].transpose(), (mean_h[:,i][np.newaxis]))

    update_phi = np.matmul((np.sum(y,axis=2)),inv(app_sum))

    z = np.zeros((100,100,len(img_train_face)))
    for i in range(len(img_train_face)):
        b = (img_train_face[i]-mean)[np.newaxis]
        h_ax = np.matmul(update_phi, mean_h[:,i])
        h_ax2 = np.matmul(h_ax[np.newaxis].transpose(),b)
        z[:,:,i] = b*b.transpose()-h_ax2

    covar1_temp = np.sum(z,axis=2)
    new_covar1 = covar1_temp/(len(img_train_face))

    phi = update_phi
    covar1 = new_covar1+(1000*np.eye(100))

    return phi, covar1

def Prob_dist(flattened_space_test, covar1, mean):

    covar1_total = np.matmul(phi,phi.transpose()) + covar1
    prob = np.zeros(len(flattened_space_test))
    for i in range(len(flattened_space_test)):
        prob[i] = np.exp(-0.5*np.matmul(np.matmul((flattened_space_test[i] - mean),(inv(covar1_total))),((flattened_space_test[i] - mean).transpose())))
    return prob

def Posterior(a, b):
    post_face = a/(a + b)
    Post_non = b/(a + b)
    return post_face, Post_non

num_it = 20
k=7

img_train_face = Data_loader_img(1)
img_train_non = Data_loader_img(2)
img_test_face = Data_loader_img(3)
img_test_non = Data_loader_img(4)

mean, covar1, phi = mean_covar1_phi(img_train_face, k)
mean_non, covar1_non, phi_non = mean_covar1_phi(img_train_non, k)

for it in range(num_it):
    if it < num_it:
        print ("it: ", it)
        mean_h, mean_h_ti = E_step( k, img_train_face, phi, mean, covar1)
        phi, covar1 = M_step(k,mean_h, mean_h_ti,mean, img_train_face )

        mean_h_non, mean_h_ti_non = E_step( k, img_train_non, phi, mean, covar1)
        phi_non, covar1_non = M_step(k,mean_h_non, mean_h_ti_non,mean_non, img_train_non )

p_ff =  Prob_dist(img_test_face, covar1, mean)
p_fn =  Prob_dist(img_test_face, covar1_non, mean_non)

p_nf =  Prob_dist(img_test_non, covar1, mean)
p_nn =  Prob_dist(img_test_non, covar1_non, mean_non)

mean_esti_face=np.reshape(mean,(10,10))
cv2.imwrite('q4_mean_face.jpg',mean_esti_face)
cv2.imshow('i',mean_esti_face)

mean_esti_face_non=np.reshape(mean_non,(10,10))
cv2.imwrite('q4_mean_non.jpg',mean_esti_face_non)
cv2.imshow('k',mean_esti_face_non)
cv2.imwrite('q4_covar_face.jpg',covar1)
cv2.imshow('k',covar1)
cv2.imwrite('q4_covar_non.jpg',covar1_non)
cv2.imshow('k',covar1_non)
post_ff, post_fn  = Posterior(p_ff, p_fn)
post_nf, post_nn  = Posterior(p_nf, p_nn)

false_positive = 0
for i in range(len(post_ff)):
    if post_nf[i]> 0.5:
        false_positive=false_positive+1
print("False Positive Rate ", float(false_positive)/100)

false_negative = 0
for i in range(len(post_ff)):
    if post_fn[i]> 0.5:
        false_negative = false_negative + 1
print("False Negative Rate ", float(false_negative)/100)

misclassification = (float(false_positive + false_negative)) / (len(post_ff) + len(post_ff))
print("Misclassification Rate", misclassification)

Posterior = np.append(   post_ff, (post_nf))
labels = np.append(np.ones(100), np.zeros(100)   )

fpr, tpr, _ = roc_curve(labels, Posterior, pos_label=1)
plt.plot(fpr, tpr)
plt.xlabel("false positive rate")
plt.ylabel("true positive rate" )
plt.show()