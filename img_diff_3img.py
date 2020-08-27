# import the necessary packages
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity
import argparse
import imutils
import cv2

#settings
#soglia tolleranza modifiche (0-255)
epsilon = 100
#area minima modifica (in pixel^2)
area_min = 100

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="first image")
ap.add_argument("-s", "--second", required=True, help="second image")
ap.add_argument("-t", "--third", required=True, help="third image")
args = vars(ap.parse_args())

# load the two input images
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
imageC = cv2.imread(args["third"])

# blur the images to remove noise
blurA = cv2.blur(imageA,(5,5))
blurB = cv2.blur(imageB,(5,5))
blurC = cv2.blur(imageC,(5,5))

# convert the images to grayscale
grayA = cv2.cvtColor(blurA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(blurB, cv2.COLOR_BGR2GRAY)
grayC = cv2.cvtColor(blurC, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
#(score, diff) = compare_ssim(grayA, grayB, full=True)
#(score, diff) = structural_similarity(grayA, grayB, full=True)
#diff = (diff * 255).astype("uint8")
#print("SSIM: {}".format(score))


(score, diff1) = structural_similarity(grayB.copy(), grayA.copy(), full=True)
diff1 = (diff1 * 255).astype("uint8")
(score, diff2) = structural_similarity(grayC.copy(), grayA.copy(), full=True)
diff2 = (diff2 * 255).astype("uint8")
(score, diff3) = structural_similarity(diff1.copy(), diff2.copy(), full=True)
diff3 = (diff3 * 255).astype("uint8")

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
#thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
thresh = cv2.threshold(diff3, epsilon, 255, cv2.THRESH_BINARY_INV)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	if (w*h) > area_min :
		cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
		#cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# show the output images
imageA = cv2.resize(imageA,(1280,720))
cv2.imshow("Original", imageA)
#imageB = cv2.resize(imageB,(1280,720))
#cv2.imshow("Modified", imageB)
diff1 = cv2.resize(diff1,(1280,720))
cv2.imshow("Diff1", diff1)
diff2 = cv2.resize(diff2,(1280,720))
cv2.imshow("Diff2", diff2)
diff3 = cv2.resize(diff3,(1280,720))
cv2.imshow("Diff3", diff3)
thresh = cv2.resize(thresh,(1280,720))
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
