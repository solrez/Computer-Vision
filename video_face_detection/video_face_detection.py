import argparse
import imutils as imt
import cv2

argp = argparse.ArgumentParser()
argp.add_argument("-f", "--face" , required=True, help="Path to the front features")
argp.add_argument("-e", "--eye"  , required=True, help="Path to the eye features")
argp.add_argument("-s", "--smile", required=True, help="Path to the smile features")
argp.add_argument("-v", "--video", help="path to the video(optional)")
args = vars(argp.parse_args())

detector1 = cv2.CascadeClassifier(args["face"])
detector2 = cv2.CascadeClassifier(args["eye"])
detector3 = cv2.CascadeClassifier(args["smile"])

if not args.get("video", False):
	camera = cv2.VideoCapture(0)

else:
	camera = cv2.VideoCapture(args["video"])

color=(10,255,15)

while True:
	(catched, fr) = camera.read()

	if args.get("video") and not catched:
		break

	fr = imt.resize(fr, width=600)
	img_gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
	rec_coor1 = detector1.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5,
		minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
	rec_coor2 = detector2.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=3,
		minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)
	rec_coor3 = detector3.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=2,
		minSize=(25, 25), flags=cv2.CASCADE_SCALE_IMAGE)

	for ((x, y, width, height),(x1,y1,width1,height1),(x2, y2, width2, height2)) in zip(rec_coor1,rec_coor2,rec_coor3):
		rect1=cv2.rectangle(fr, (x, y), (x + width, y + height), color, 2)
		rect2=cv2.rectangle(fr, (x1, y1), (x1 + width1, y1 + height1), color, 2)
		rect3=cv2.rectangle(fr, (x2, y2), (x2 + width2, y2 + height2), color, 2)
		cv2.putText(rect1, 'Face', (x, y-12), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)
		cv2.putText(rect2, 'Eye', (x1, y1-12), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)
		cv2.putText(rect3, 'Lip', (x2, y2-12), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)

	cv2.imshow("Press e to exit", fr)

	if cv2.waitKey(1) == ord("e"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()