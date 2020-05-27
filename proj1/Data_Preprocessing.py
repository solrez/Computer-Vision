import glob
import cv2
import linecache
import shutil
import random
import pdb
from matplotlib import pyplot as plt
def overlap_calculator(ba, bb):

    x1 = max(ba[0], bb[0])
    y1 = max(ba[1], bb[1])
    x2 = min(ba[2], bb[2])
    y3 = min(ba[3], bb[3])

    intersection_sur = (x2 - x1) * (y2 - y1)
    surface_aa = (ba[2] - ba[0]) * (ba[3] - ba[1])
    surface_bb = (bb[2] - bb[0]) * (bb[3] - bb[1])
    ro = intersection_sur / float(surface_aa + surface_bb - intersection_sur)
    if ro >.5:
        con=True
        return con
    else:
        con =False
        return con

    return con

# def overlap_condition(co):
#     if co < 0.3:
#         return False
#     else: return True

dir = "C:/Users/Solei/OneDrive/Desktop/rsoleim_project01/FDDB-folds/" 
image_all = "C:/Users/Solei/OneDrive/Desktop/rsoleim_project01/images/"  
image_face ="C:/Users/Solei/OneDrive/Desktop/rsoleim_project01/face/Positive/" 
image_nonface = "C:/Users/Solei/OneDrive/Desktop/rsoleim_project01/face/Negative/"
im_format = ".jpg"
number = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
count = 118
non_face_im_size = 60
for num in number:
    filename_ellipse = dir+"FDDB-fold-"+num+"-ellipseList.txt"
    filename_images = dir+"FDDB-fold-"+num+".txt"
    file_images = open(filename_images,'r')
    image_names = file_images.readlines()

    for image_name in image_names:
        with open(filename_ellipse,'r') as file_ellipse:
            for num, line in enumerate(file_ellipse, 1):
                if image_name in line:
                    num_face_line = linecache.getline(filename_ellipse, (num+1))
                    if num_face_line.strip() == "1":
                        ellipse_dim = (linecache.getline(filename_ellipse, (num+2))).split()
                        image = cv2.imread(image_all+image_name.strip()+im_format)
                        x1 = int(float(ellipse_dim[3])-float(ellipse_dim[1]))            
                        x2 = int(float(ellipse_dim[3])+float(ellipse_dim[1]))            
                        y1 = int(float(ellipse_dim[4])-float(ellipse_dim[0]))            
                        y2 = int(float(ellipse_dim[4])+float(ellipse_dim[0]))            
                        im_cropped = image[y1:y2,x1:x2]
                        if (im_cropped.shape[0]) and (im_cropped.shape[1]):
                            count = count+1
                            im_cropped = cv2.resize(im_cropped, (60,60))
                            # gray_im = cv2.cvtColor(im_cropped, cv2.COLOR_RGB2GRAY )
                            cv2.imwrite(image_face+str(count)+im_format, im_cropped)
                        condition = True
                        loop_count = 0
                        while condition:                                                       
                            rand_crop_x = random.randint(0,image.shape[0]-60)                   
                            rand_crop_y = random.randint(0,image.shape[1]-60)
                            x2_non = rand_crop_x+non_face_im_size
                            y2_non = rand_crop_y+non_face_im_size
                            ba = [x1,y1,x2,y2]
                            bb = [rand_crop_x,x2_non,rand_crop_y,y2_non]
                            condition = overlap_calculator(ba,bb)
                            loop_count = loop_count+1
                            non_face_im=image[rand_crop_x:x2_non,rand_crop_y:y2_non]                               
                        if (non_face_im.shape[0]) and (non_face_im.shape[1]):
                            count = count+1
                            # non_face_im = cv2.cvtColor(non_face_im, cv2.COLOR_RGB2GRAY)
                            non_face_im = cv2.resize(non_face_im, (60, 60))
                            cv2.imwrite(image_nonface+str(count)+im_format, non_face_im)
                            


