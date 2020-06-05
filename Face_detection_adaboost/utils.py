
fl_T   = True
fl_B = True


face_dir    = "./test_train_img/face/"
nonface_dir = "./test_train_img/nonface/"


face_dir_test        = "./test_train_img/facetest/"
nonface_dir_test     = "./test_train_img/nonfacetest/"

feature_dir = "./features/features_train.cache"
feature_dir_test="./features/features_test.cache"
model_dir = "./model/adaboost_classifier.cache0"

fig_dir = "./figure/"

img_x_size = 19
img_y_size  = 19

feature_num = 37862

face_samples     = 2000
nonface_samples     = 2000

face_samples_test = 20
nonface_samples_test = 20

face_label = +1
nonface_label = -1

wh = 255
bl = 0

tpr_final = 0.999
fpr_final = 0.0005

haar_type1 = "I"
haar_type2 = "II"
haar_type3 = "III"
haar_type4 = "IV"
haar_type5 = "V"

num_weak_class = 110

