
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


def mean_cov(n_mixture):

    k_n = np.ones(n_mixture)/n_mixture
    mean = np.random.randint(0,255, size=(n_mixture, 100))
    covar1=np.zeros((100,100,n_mixture))
    for i in range(n_mixture):
        for j in range(100):
            for k in range(100):
                if j==k:
                    covar1[j][k][i]=int((np.random.random())+(1000*(2)))
                else:
                    covar1[j][k][i]=0
    return k_n, mean, covar1


def E_step( n_mixture, img_train_face, k_n, mean, covar1):
    prob = np.zeros((len(img_train_face), n_mixture))
    sum_prob = np.zeros(len(img_train_face))
    for i in range (len(img_train_face)):
        for k in range (n_mixture):
            x = np.matmul(np.matmul((img_train_face[i] - mean[k]),(inv(covar1[:,:,k]))),((img_train_face[i] - mean[k]).transpose()))
            prob[i][k] =  k_n[k]*np.exp(-0.5*x)
        sum_prob[i] = np.sum(prob[i])

    hi = np.zeros((len(img_train_face), n_mixture))
    for i in range(len(img_train_face)):
        for j in range(n_mixture):
            hi[i][j] = prob[i][j]/sum_prob[i]

    in_t = np.zeros(n_mixture)
    for i in range(n_mixture):
        in_t[i]=np.sum(hi[:,i])
    return hi, in_t

def M_step(n_mixture, r, in_t, img_train_face ):

    new_k_n = np.zeros(n_mixture)
    for i in range(n_mixture):
        new_k_n[i] = in_t[i]/np.sum(in_t)

    r_t=r.transpose()
    mean_update = np.zeros((n_mixture, 100))
    mean_update_hi = np.zeros((len(img_train_face),100))
    for i in range (n_mixture):
        for j in range(len(img_train_face)):
            for k in range(100):
                mean_update_hi[j][k] = r_t[i][j]*img_train_face[j][k]

        for j in range(100):
           mean_update[i][j] = np.sum(mean_update_hi[:,j])

    cov_update=np.zeros((100,100,n_mixture))
    for i in range(n_mixture):
        f1,f2,f3,f4 = np.zeros((len(img_train_face),100)),np.zeros((len(img_train_face),100)),np.zeros((100,1)),np.zeros((100,1))
        for j in range(len(img_train_face)):
            f1[j] = (img_train_face[j] - mean_update[i])
            f2[j] = f1[j]*r[j][i]
        for k in range(100):
            f3[k] = np.sum(f1[:,k])
            f4[k] = np.sum(f2[:,k])
        cov_update_hi=np.matmul(f3,(f4).transpose())
        cov_update[:,:,i]=cov_update_hi/np.sum(in_t[i])

    covar11=np.zeros((100, 100,n_mixture))
    for j in range(n_mixture):
        for i in range(100):
            for k in range(100):
                if i==k:
                    covar11[i][k][j] = (((cov_update[i][k][j] - np.min(cov_update[:,:,j]))/ (np.max(cov_update[:,:,j])-np.min(cov_update[:,:,j])))+10*1000)
                else:
                    covar11[i][k][j] = 0
    mean1=np.zeros((n_mixture, 100))
    for j in range(n_mixture):
        for i in range(100):
            mean1[j][i] = ((mean_update[j][i] - np.min(mean_update[j,:]))/ (np.max(mean_update[j,:])-np.min(mean_update[j,:])))*255#

    k_n = new_k_n
    mean = mean1
    covar1 = covar11

    return new_k_n, mean1, covar11

def Gauss_mixture_dist( n_mixture, img_vec, k_n, covar1, mean):

    prob = np.zeros((len(img_vec), n_mixture))
    sum_prob = np.zeros(len(img_vec))
    for i in range (len(img_vec)):
        for k in range (n_mixture):
            y = np.matmul(np.matmul((img_vec[i] - mean[k]),(inv(covar1[:,:,k]))),((img_vec[i] - mean[k]).transpose()))
            prob[i][k] =  k_n[k]*np.exp(-0.5*y)
        sum_prob[i] = np.sum(prob[i])
    return  sum_prob

def Posterior(a, b):
    post_face = a/(a + b)
    Post_non = b/(a + b)
    return post_face, Post_non

n_mixture = 4
num_it = 10
img_train_face = Data_loader_img(1)
img_train_non = Data_loader_img(2)
img_test_face = Data_loader_img(3)
img_test_non = Data_loader_img(4)

k_n_face, mean_face, covar1_face = mean_cov(n_mixture)
k_n_non, mean_non, covar1_non = mean_cov(n_mixture)

for it in range(num_it):
    if it < num_it:
        print( "it: ", it)
        hi_face, hi_h_face = E_step( n_mixture, img_train_face, k_n_face, mean_face, covar1_face)
        k_n_face, mean_face, covar1_face = M_step(n_mixture, hi_face, hi_h_face, img_train_face )

        hi_non, hi_h_non = E_step( n_mixture, img_train_non, k_n_non, mean_non, covar1_non)
        k_n_non, mean_non, covar1_non = M_step(n_mixture, hi_non, hi_h_non, img_train_non )

for i in range(n_mixture):
    mean_esti_face=np.reshape(mean_face[i],(10,10))
    cv2.imwrite('q2_mean_face_'+str(i)+'.jpg',mean_esti_face)
    cv2.imshow(str(i),mean_esti_face)
for i in range(n_mixture):
    mean_esti_non=np.reshape(mean_non[i],(10,10))
    cv2.imwrite('q2_mean_non_'+str(i)+'.jpg',mean_esti_non)
    cv2.imshow(str(i),mean_esti_non)

for i in range(n_mixture):
    cv2.imwrite('q2_covar_face_'+str(i)+'.jpg',covar1_face[:,:,i])
    cv2.imshow(str(i),covar1_face[:,:,i])

for i in range(n_mixture):
    cv2.imwrite('q2_covar_non_'+str(i)+'.jpg',covar1_non[:,:,i])
    cv2.imshow(str(i),covar1_non[:,:,i])

p_ff = Gauss_mixture_dist(n_mixture, img_test_face, k_n_face, covar1_face, mean_face)
p_nf = Gauss_mixture_dist(n_mixture, img_test_non, k_n_face, covar1_face, mean_face)

p_fn = Gauss_mixture_dist(n_mixture, img_test_face, k_n_non, covar1_non, mean_non)
p_nn = Gauss_mixture_dist(n_mixture, img_test_non, k_n_non, covar1_non, mean_non)

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
