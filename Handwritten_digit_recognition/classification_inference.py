from __future__ import print_function
from sklearn.externals import joblib as jb
import argparse
import mahotas as ms
import cv2
import mahotas
import utils

argp = argparse.ArgumentParser()
argp.add_argument("-m", "--model", required=True, help="model directory")
argp.add_argument("-i", "--img", required= True, help="image directory")
args = vars(argp.parse_args())

model = jb.load(args["model"])

img = cv2.imread(args["img"])
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
img_corner = cv2.Canny(img_blur, 30, 150)

(contours, _) = cv2.findContours(img_corner.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted([(cur, cv2.boundingRect(cur)[0]) for cur in contours], key=lambda x: x[1])

for (cur, _) in contours:
	(x, y, width, height) = cv2.boundingRect(cur)

	if width >= 8 and height >= 19:
		tmp = img_gray[y:y + height, x:x + width]
		cut_img = tmp.copy()
		h_th = ms.thresholding.otsu(tmp)
		cut_img[cut_img > h_th] = 255
		cut_img = cv2.bitwise_not(cut_img)
		cut_img = utils.norm_img(cut_img, 20)
		cut_img = utils.mass_center(cut_img, (20, 20))

		dat = utils.hog(cut_img,orientations=18, pixelsPerCell=(5, 5), cellsPerBlock=(1, 1), normalize=True)
		Number = model.predict(dat.reshape(1, -1))[0]

		cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 1)
		cv2.putText(img, str(Number), (x - 10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 0), 2)
		cv2.imshow("img", img)
		cv2.waitKey(0)