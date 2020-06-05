from skimage import feature
import numpy as np
import mahotas as ms
import imutils as imt
import cv2

def hog(img,orientations=12, pixelsPerCell=(4, 4), cellsPerBlock=(2, 2), normalize=True):
	dat = feature.hog(img, orientations=orientations, pixels_per_cell=pixelsPerCell,
			cells_per_block=cellsPerBlock, transform_sqrt=normalize)
	dat[dat < 0] = 0
	return dat

def getting_data(path):
	data = np.genfromtxt(path, delimiter=",", dtype="uint8")
	labels = data[:, 0]
	data = data[:, 1:].reshape(data.shape[0], 28, 28)
	return data, labels

def norm_img(img, img_width):
	(height, width) = img.shape[:2]
	moments = cv2.moments(img)
	skwidth_img = moments["mu11"] / moments["mu02"]
	tmp = np.float32([ [1, skwidth_img, -0.5 * width * skwidth_img], [0, 1, 0]])
	img = cv2.warpAffine(img, tmp, (width, height), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	img = imt.resize(img, width=img_width)
	return img

def mass_center(img, dim):
	(width, height) = dim
	if img.shape[1] > img.shape[0]:
		img = imt.resize(img, width=width)
	else:
		img = imt.resize(img, height=height)

	output = np.zeros((height, width), dtype="uint8")
	dist_x = (width - img.shape[1]) // 2
	dist_y = (height - img.shape[0]) // 2
	output[dist_y:dist_y + img.shape[0], dist_x:dist_x + img.shape[1]] = img

	(cen_y, cen_x) = np.round(ms.center_of_mass(output)).astype("int32")
	(cor_x, cor_y) = ((dim[0] // 2) - cen_x, (dim[1] // 2) - cen_y)
	tmp = np.float32([[1, 0, cor_x], [0, 1, cor_y]])
	output = cv2.warpAffine(output, tmp, dim)
	return output