import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import roc_curve
from scipy.special import digamma
from scipy.special import gamma
import scipy
from scipy import optimize
import pdb

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

def mean_cov_D_nu(img_train_face, n_mixture):
    k_n = np.zeros(n_mixture)
    nu = np.random.randint(100,200, size=n_mixture)#
    mean = np.random.randint(0,255, size=(n_mixture, 100))
    covar1=np.zeros((100,100,n_mixture))
    for i in range(n_mixture):
        for j in range(100):
            for k in range(100):
                if j==k:
                    covar1[j][k][i]=int((np.random.random())+(1000))
                else:
                    covar1[j][k][i]=0

    return mean, covar1, k_n, nu

def E_step( n_mixture, img_train_face, D, mean, covar1, nu):
    mean_h = np.zeros((len(img_train_face),n_mixture))
    mean_hi = np.zeros((len(img_train_face),n_mixture))
    y = np.zeros((len(img_train_face),n_mixture))
    h_res = np.zeros((len(img_train_face),n_mixture))
    ax_vec = np.zeros((len(img_train_face), n_mixture))
    D=100
    for i in range(len(img_train_face)):
        for j in range(n_mixture):
            ax_vec[i][j] = np.matmul(np.matmul((img_train_face[i] - mean[j,:]),(inv(covar1[:,:,j]))),((img_train_face[i] - mean[j,:]).transpose()))
            mean_h[i][j]=(nu[j]+D)/(nu[j]+ax_vec[i][j])
            mean_hi[i][j] = digamma((nu[j]/2 + D/2), out=None) - np.log10((nu[j]/2) + (ax_vec[i][j]/2))
            y[i][j] = (gamma((nu[j]+D)/2) * ((1 + (ax_vec[i][j]/nu[j]))**(-(nu[j]+D)/2)))/gamma(nu[j]/2)
            SM = np.sum(y[i])
        for j in range(n_mixture):
            h_res[i][j] = y[i][j]/SM

    return mean_h, mean_hi, h_res


def M_step(n_mixture, mean_h, mean_hi, h_res, img_train_face, nu, D ):
    new_k_n = (np.sum(h_res, axis=0))/len(img_train_face)

    x = np.sum(mean_h, axis=0)
    flattened_array = np.asarray(img_train_face)
    y=np.zeros((len(img_train_face),D,n_mixture))
    mean_update = np.zeros((n_mixture,D))
    for j in range(n_mixture):
        for i in range(len(img_train_face)):
             y[i,:,j] = mean_h[i,j]*flattened_array[i]
        y1 = np.sum(y, axis=0)
        mean_update = y1/x[j]
        mean_update = mean_update.transpose()
    #
    covar1_update=np.zeros((D,D,n_mixture))
    for i in range(n_mixture):
        f1,f2,f3,f4 = np.zeros((len(img_train_face),D)),np.zeros((len(img_train_face),D)),np.zeros((D,1)),np.zeros((D,1))
        for j in range(len(img_train_face)):
            f1[j] = (flattened_array[j] - mean_update[i,:])
            f2[j] = f1[j,:]*mean_h[j][i]
        for k in range(D):
            f3[k] = np.sum(f1[:,k])
            f4[k] = np.sum(f2[:,k])
        covar1_update_hi=np.matmul(f3,(f4).transpose())
        covar1_update[:,:,i]=covar1_update_hi/x[i]

    covar11=np.zeros((100, 100, n_mixture))
    for j in range(n_mixture):
        for i in range(100):
            for k in range(100):
                    covar11[i][k][j] = (((covar1_update[i][k][j] - np.min(covar1_update[:,:,j]))/ (np.max(covar1_update[:,:,j])-np.min(covar1_update[:,:,j]))))
        covar11[:,:,j] = covar11[:,:,j]+(100*np.eye(100))
    nu_update=np.zeros(n_mixture)
    for i in range(n_mixture):
        def f(nu):
            prob = (len(img_train_face) * ((nu/2)* np.log10(nu/2))) + (len(img_train_face)* np.log10(gamma(nu/2))) - (((nu/2)-1)* np.sum(mean_hi[:,i])) + ((nu/2)*np.sum(mean_h[:,i]))
            return prob
        nu_opt = scipy.optimize.fmin(f, nu[i])
        nu_update[i] = nu_opt[0]

    k_n = new_k_n
    mean = mean_update
    covar1 = covar11

    return k_n, mean, covar1,nu_update


def prob_dist(img_train_face, covar1, mean, n_mixture, nu):
    
    mean_h = np.zeros((len(img_train_face),n_mixture))
    mean_hi = np.zeros((len(img_train_face),n_mixture))
    y = np.zeros((len(img_train_face),n_mixture))
    h_res = np.zeros((len(img_train_face),n_mixture))
    x = np.zeros((len(img_train_face), n_mixture))
    ins_p =  np.zeros((len(img_train_face), n_mixture))
    prob = np.zeros(len(img_train_face))
    D=100
    for i in range(len(img_train_face)):
        for j in range(n_mixture):
            x[i][j] = np.matmul(np.matmul((img_train_face[i] - mean[j,:]),(inv(covar1[:,:,j]))),((img_train_face[i] - mean[j,:]).transpose()))
            mean_h[i][j]=(nu[j]+D)/(nu[j]+x[i][j])
            mean_hi[i][j] = digamma((nu[j]/2 + D/2), out=None) - np.log10((nu[j]/2) + (x[i][j]/2))
            y[i][j] = (gamma((nu[j]+D)/2) * ((1 + (x[i][j]/nu[j]))**(-(nu[j]+D)/2)))/gamma(nu[j]/2)
            SM = np.sum(y[i])
        for j in range(n_mixture):
            h_res[i][j] = y[i][j]/SM
            ins_p[i][j] =  k_n[j]*h_res[i][j]
        prob[i] = np.sum(ins_p[i])


    return prob

def Posterior(a, b):
    post_face = a/(a + b)
    Post_non = b/(a + b)
    return post_face, Post_non

num_it = 20
n_mixture =2
D = 100

img_train_face = Data_loader_img(1)
img_train_non = Data_loader_img(2)
img_test_face = Data_loader_img(3)
img_test_non = Data_loader_img(4)

mean, covar1, k_n, nu  = mean_cov_D_nu(img_train_face, n_mixture)
mean_non, covar1_non, k_n_non, nu_non  = mean_cov_D_nu(img_train_non, n_mixture)

for it in range(num_it):
    if it < num_it:
        print ("it: ", it)
        mean_h, mean_hi, h_res = E_step( n_mixture, img_train_face, D, mean, covar1, nu)
        #pdb.set_trace()
        k_n, mean, covar1,nu = M_step(n_mixture, mean_h, mean_hi, h_res, img_train_face, nu, D)
        #print(nu)
        mean_h_non, mean_hi_non, h_res_non = E_step( n_mixture, img_train_non, D, mean_non, covar1_non, nu_non)
        k_n_non, mean_non, covar1_non,nu_non = M_step(n_mixture, mean_h_non, mean_hi_non, h_res_non, img_train_face, nu_non, D)
        #print(nu_non)
for i in range(n_mixture):
    mean_esti_face=np.reshape(mean[i,:],(10,10))
    cv2.imwrite('q5_mean_face_mix_t_distribution_'+str(i)+'.jpg',mean_esti_face)
    cv2.imshow('k',mean_esti_face)
for i in range(n_mixture):
    mean_esti_face_non=np.reshape(mean_non[i,:],(10,10))
    cv2.imwrite('q5_mean_non_face_mix_t_distribution'+str(i)+'.jpg',mean_esti_face_non)
    cv2.imshow('h',mean_esti_face_non)

for i in range(n_mixture):
    cv2.imwrite('q5_covar_face_mix_t_distribution'+str(i)+'.jpg',covar1[:,:,i])
    cv2.imshow('z',covar1[:,:,i])

for i in range(n_mixture):
    cv2.imwrite('q5_mean_non_mix_t_distribution'+str(i)+'.jpg',covar1_non[:,:,i])
    cv2.imshow('f',covar1_non[:,:,i])

p_ff =  prob_dist(img_test_face, covar1, mean, n_mixture, nu)
p_fn =  prob_dist(img_test_face, covar1_non, mean_non, n_mixture, nu_non)

p_nf =  prob_dist(img_test_non, covar1, mean, n_mixture, nu)
p_nn =  prob_dist(img_test_non, covar1_non, mean_non, n_mixture, nu_non)


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

Posterior = np.append(   post_ff, (post_nf))
labels = np.append(np.ones(100), np.zeros(100))

fpr, tpr, _ = roc_curve(labels, Posterior, pos_label=1)
plt.plot(fpr, tpr, color='darkorange')
plt.xlabel("false positive rate")
plt.ylabel("true positive rate" )
plt.show()
