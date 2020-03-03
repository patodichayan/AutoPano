#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
# Add any python libraries here



def main():
	# Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    
    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

	"""
	Read a set of images for Panorama stitching
	"""
	im = cv2.imread("../Data/Train/1.jpg")
	im2 = cv2.resize(im, (320, 240))
	print(im2.shape)
	cv2.imshow('og', im)
	cv2.imshow('resize', im2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	"""
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""
	
	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

    
if __name__ == '__main__':
    main()
 





	else : 
		Img1_full = []
		Patches_ = []
		Corners_ = []
		I2_ = []

		while ImageNum < MiniBatchSize:
			# Generate random image

			if(validation):
				RandIdx = random.randint(1, 1000)
				RandImageName = BasePath + "Val/" + str(RandIdx) + '.jpg'
			else:
				RandIdx = random.randint(0, len(DirNamesTrain)-1)
				RandImageName = BasePath + DirNamesTrain[RandIdx] + '.jpg'   
			ImageNum += 1

			##########################################################
			# Add any standardization or data augmentation here!
			##########################################################
			I1 = np.float32(cv2.imread(RandImageName))

			# for each image generate 10 corresponding images

			# patch size
			patch_width = 128
			patch_height = 128

			# perturbation factor p 
			p = 32

			# copy of img
			im = copy.deepcopy(I1)

			# image pre-processing
			if(len(im.shape)==3):
			    im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
			im = cv2.resize(im, (320, 240))
			im=(im-np.mean(im))/255

			# origin of patch A
			ox = random.randrange(p, im.shape[1] - p - patch_width - 20)
			oy = random.randrange(p, im.shape[0] - p - patch_height - 20) 

			patch_a = np.array([[ox, oy], 
			                    [ox+patch_width, oy], 
			                    [ox+patch_width, oy+patch_height], 
			                    [ox, oy+patch_height]], np.float32)
			patch_b = []

			# perform perturbations on patch a points
			for pt in patch_a:
			    patch_b.append([pt[0]+random.randrange(-p, p), pt[1]+random.randrange(-p, p)])
			patch_b = np.array(patch_b, np.float32)

			# compute homography
			H_ab = cv2.getPerspectiveTransform(patch_a, patch_b)
			H_ba = np.linalg.inv(H_ab) 

			# warp the image
			warp_img = cv2.warpPerspective(im, H_ba, (im.shape[1], im.shape[0]))

			# extract the patches
			cropped_a = im[oy:oy+patch_height, ox:ox+patch_width]
			cropped_b = warp_img[oy:oy+patch_height, ox:ox+patch_width]

			# iniitialize the model params
			input_data = np.dstack((cropped_a, cropped_b))


			Img1_full.append(np.float32(img))
			Patches_.append(input_data)
			Corners_.append(np.float32(patch_a))
			I2_.append(np.float32(cropped_b.reshape(128,128,1)))

	        # Label = convertToOneHot(TrainLabels[RandIdx], 10)
		return Img1_full, Patches_, Corners_, I2_
