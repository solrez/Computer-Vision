from sklearn.externals import joblib as jb
from sklearn.svm import LinearSVC
import utils
import argparse

argp = argparse.ArgumentParser()
argp.add_argument("-d", "--dataset", required=True, help="dataset file")
argp.add_argument("-m", "--model", required=True, help="Path model being saved")
args = vars(argp.parse_args())

Numbers, Labels =utils.getting_data(args["dataset"])
data = []

for image in Numbers:
	image = utils.norm_img(image, 20)
	image = utils.mass_center(image, (20, 20))
	dat = utils.hog(image,orientations=18, pixelsPerCell=(5, 5), cellsPerBlock=(1, 1), normalize=True)
	data.append(dat)

model = LinearSVC(random_state=40)
model.fit(data, Labels)
jb.dump(model, args["model"])