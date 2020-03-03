#!/usr/bin/evn python

"""
CMSC733 Spring 2020: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chayan Kumar Patodi (ckp1804@terpmail.umd.edu)
University of Maryland, College Park

Saket Seshadri Gudimetla Hanumath (saketsgh@terpmail.umd.edu)
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
from glob import glob
import argparse
import os
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import random
import copy

# Add any python libraries here

#Helping Functions.

def load_images(indices):
	images = []
	for i in indices:
		images.append((plt.imread(i)))

	return images

def get_corner_harris(img):

	copy_img = copy.deepcopy(img)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#gray = cv2.GaussianBlur(gray,(5,5),0)
	dst = cv2.cornerHarris(gray, 8, 3, 0.04)

	# result is dilated for marking the corners, not important
	dst = cv2.dilate(dst, None)

	# threshold
	dst[dst < 0.01 * dst.max()] = 0

	# Threshold for an optimal value, it may vary depending on the image.
	copy_img[dst > 0.01 * dst.max()] = [255, 0, 0]

	plt.figure()
	plt.imshow(copy_img)
	plt.show()

	return copy_img, dst

def ANMS(C_img, N_best,img):
	# C_img is the corner score image for which we find local maxima
	image = copy.deepcopy(img)
	coordinates = peak_local_max(C_img, min_distance=20)

	N_strong = coordinates.shape[0]
	r = np.array([np.inf for i in range(N_strong)])
	x = np.zeros((N_strong,1))
	y = np.zeros((N_strong,1))
	ED = 0

	for i in range(N_strong):
		for j in range(N_strong):
			xj, yj = coordinates[j][0], coordinates[j][1]
			xi, yi = coordinates[i][0], coordinates[i][1]
			if (C_img[xj, yj] > C_img[xi, yi]):
				ED = (xi - xj) ** 2 + (yi - yj) ** 2
			if (ED < r[i]):
				r[i] = ED
				x[i] = xi
				y[i] = yi

	r_sort = np.argsort(r)

	# pick top N_best points
	r_sort = r_sort[:N_best]
	
	coord = []

	for i in r_sort:
		x, y = coordinates[i][0], coordinates[i][1]
		coord.append((x, y))
		cv2.circle(image,(coordinates[i][1],coordinates[i][0]),3,255,-1)

	plt.imshow(image)
	plt.show()
        
	return coord

def feature_desc(coordinates, anms_op):
	vec_list = []
	image = copy.deepcopy(anms_op)
	val = []
	count = 0
	for c in coordinates:
		

		i_c, j_c = c[0], c[1]
		
		patch = image[int(i_c)-20:int(i_c)+20, int(j_c)-20:int(j_c)+20]
		
		# blur image
		patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

		patch_blur = cv2.GaussianBlur(patch,(5,5),0)

		# resize to 8X8
		patch_blur = cv2.resize(patch_blur, (8, 8),interpolation = cv2.INTER_AREA)
		val.append(patch_blur)				

		patch_vec = np.reshape(patch_blur, (64, 1))

		# standardize the vec
		mean = np.mean(patch_vec)
		std = np.std(patch_vec)

		patch_vec = (patch_vec - mean)/(std+10**-8)
		vec_list.append(patch_vec)

	return vec_list

def feature_match(vec_list1,vec_list2):
	locations = []
	matching_ratio = 0.90
	for ind_v1,v1 in enumerate(vec_list1):
		feat_confidence = []
		for v2 in vec_list2:
			feat_confidence.append(sum((v1-v2)**2))
		

		#index = np.argsort((feat_confidence))
		ind_v2 = sorted(range(len(feat_confidence)), key=feat_confidence.__getitem__)
		ratio = feat_confidence[ind_v2[0]]/feat_confidence[ind_v2[1]]

		if ratio < matching_ratio:

			locations.append([ind_v1,ind_v2[0]])

	return locations

#Plotting Functions.

def plot_match(coordinates1,coordinates2,locations,image1,image2):

	img1 = copy.deepcopy(image1)
	img2 = copy.deepcopy(image2)
	
	coordinates1 = np.asarray(coordinates1)
	coordinates2 = np.asarray(coordinates2)
	
	corner1_keypoints = []
	for cornerInd in range(coordinates1.shape[0]):
		corner1_keypoints.append(cv2.KeyPoint(coordinates1[cornerInd,1], coordinates1[cornerInd,0], 5))

	corner2_keypoints = []
	for cornerInd in range(coordinates2.shape[0]):
		corner2_keypoints.append(cv2.KeyPoint(coordinates2[cornerInd,1], coordinates2[cornerInd,0], 5))
	
	matchesImg = np.array([])
	dmatchvec= []
	for m in locations:

		dmatchvec.append(cv2.DMatch(m[0],m[1],1))

	draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0))
	matchesImg = cv2.drawMatches(img1,corner1_keypoints,img2,corner2_keypoints,dmatchvec,matchesImg,**draw_params)
	plt.imshow(matchesImg)
	plt.show()

def show_correspondence(imgA, imgB, X1, Y1, X2, Y2):
	
	height = max(imgA.shape[0],imgB.shape[0])
	width = imgA.shape[1] + imgB.shape[1]
	clubimage = np.zeros((height,width,3),type(imgA.flat[0]))
	clubimage[:imgA.shape[0],:imgA.shape[1],:] = imgA
	clubimage[:imgB.shape[0],imgA.shape[1]:,:] = imgB
	shiftX = imgA.shape[1]
	for i in xrange(X1.shape[0]):
		cv2.circle(clubimage, (X1[i], Y1[i]), 3, (255,0,0), -1)
		cv2.circle(clubimage, (X2[i]+shiftX, Y2[i]), 3, (255,0,0), -1)
		cv2.line(clubimage, (X1[i], Y1[i]), (X2[i]+shiftX, Y2[i]), (0,255,0), 1)

	plt.imshow(clubimage)
	plt.show()

#Homography , Ransac and Stitching.

def ransac_homog(locations, coordinates1,coordinates2,N_iter, thresh,image1,image2):

	pts_img1 = []
	pts_img2 = []
	for locs in locations:
		x1,y1 = coordinates1[locs[0]]
		x2,y2 = coordinates2[locs[1]]
		pts_img1.append([y1,x1])
		pts_img2.append([y2,x2])

	# Calculate homography via RANSAC in following steps 

	inliers = 0
	max_inliers = 0
	iterations = 0
	flag = 0

	inlier_lim = min(0.3*len(pts_img1),0.3*len(pts_img2))
	
	# make homogenuous
	pts_img1_arr = np.array(pts_img1)
	pts_img1_arr = np.append(pts_img1_arr, np.ones((pts_img1_arr.shape[0], 1)), axis=1)
	pts_img2_arr = np.array(pts_img2)

	inlier_locs = []

	while(iterations < N_iter):
	    
	    iterations += 1
	    # generate 4 random points from img 1 and 2
	    # **if you don't put np.float32 then error is generated stating pts are of not the form cv::Umat()
	    p1 = np.array(random.sample(pts_img1, 4), np.float32)
	    p2 = np.array(random.sample(pts_img2, 4), np.float32)
	    
	    #  compute homography between them
	    homography= cv2.getPerspectiveTransform(p1, p2)

	    # shape is 4X3 so transpose it so that matrix multiplication is possible
	    # compute SSD between actual points and warped points
	    pts_warped = np.dot(homography, np.transpose(pts_img1_arr))
	    pts_warped = np.transpose(pts_warped)
	    pts_warped = np.where(pts_warped == 0, 0.001, pts_warped)
	    
	    # converting them back to cartesian
	    pts_warped[:, 0] = pts_warped[:, 0]/pts_warped[:, 2]
	    pts_warped[:, 1] = pts_warped[:, 1]/pts_warped[:, 2]
	    pts_warped = pts_warped[:, 0:2]
		# ssd = np.array((pts_img1_arr.shape[0], pts_img1_arr.shape[1]))

	    # calc diff, then square it and then take sum
	    ssd = pts_img2_arr - pts_warped
	    ssd = np.square(ssd)
	    
	    # removing homogenuous coordinate
	    ssd = np.sum(ssd, axis=1)

	    # comparing with threshold
	    inliers = (np.where(ssd<thresh)[0])
	    
	    if len(inliers) > max_inliers:
	        
	        max_inliers = len(inliers)
	        inlier_locs = inliers
	            
	print("iterations --> {}".format(iterations))
	print("inliers --> {}".format(max_inliers))

	bm_img1 = []
	bm_img2 = []

	matches_inliers = []
	for l in inlier_locs:

		p1 = pts_img1[l]
		p2 = pts_img2[l]
		bm_img1.append(p1)
		bm_img2.append(p2)
		matches_inliers.append([l,l])
	
	
	if max_inliers <  min(inlier_lim,50):
		print("Not enough matches")
		flag = 1

	return bm_img1, bm_img2, matches_inliers , flag

def homography(inliers_src, inliers_dst):
	
	M = []
	for i in range(len(inliers_src)):
		M.append([-inliers_src[i][0], -inliers_src[i][1], -1, 0, 0 ,0, inliers_src[i][0]*inliers_dst[i][0], inliers_src[i][1]*inliers_dst[i][0], inliers_dst[i][0]])
		M.append([ 0, 0 ,0,-inliers_src[i][0], -inliers_src[i][1], -1, inliers_src[i][0]*inliers_dst[i][1], inliers_src[i][1]*inliers_dst[i][1], inliers_dst[i][1]])
	s, v, vh = np.linalg.svd(M)
	H = vh[-1,:]
	
	return  (H.reshape((3,3)))

def combine(img1, img2, H):

	'''warp img2 to img1 with homograph H'''
	h1,w1 = img1.shape[:2]
	h2,w2 = img2.shape[:2]

	pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
	pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
	pts2_ = cv2.perspectiveTransform(pts2, H)
	pts = np.concatenate((pts1, pts2_), axis=0)
	[xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
	[xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
	t = [-xmin,-ymin]
	Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
	result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
	result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1

	return result

def main():
	# Add any Command Line arguments here
 
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--BasePath', default="../Data/Train/Set1", help='Folder of Test Images')

	Args = Parser.parse_args()
	BasePath = Args.BasePath

	path = str(BasePath) + str("/*.jpg")

	"""
	Read a set of images for Panorama stitching
	"""

	ind = sorted(glob(path))
	images = load_images(ind)

	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""

	for i in range(len(images)-1):
		corners1, CornerScoreImg1 = get_corner_harris(images[i])
		corners2, CornerScoreImg2 = get_corner_harris(images[i+1])

		"""
		Perform ANMS: Adaptive Non-Maximal Suppression
		Save ANMS output as anms.png
		"""
		
		coordinates1 = ANMS(CornerScoreImg1,600,images[i])
		coordinates2 = ANMS(CornerScoreImg2,600,images[i+1])

		"""
		Feature Descriptors
		Save Feature Descriptor output as FD.png
		"""
		
		vec_list1 = feature_desc(coordinates1,images[i])
		vec_list2 = feature_desc(coordinates2,images[i+1])

		"""
		Feature Matching
		Save Feature Matching output as matching.png
		"""

		locations = feature_match(vec_list1,vec_list2)
		plot_match(coordinates1,coordinates2,locations,images[i],images[i+1])

		"""
		Refine: RANSAC, Estimate Homography
		"""

		inliers_src,inliers_dst,inlier_locs ,flag = ransac_homog(locations,coordinates1,coordinates2,700000,1000,images[i],images[i+1])
		
		x1 = np.asarray(list(list(zip(*inliers_src))[0])) 
		y1 = np.asarray(list(list(zip(*inliers_src))[1]))
		x2 = np.asarray(list(list(zip(*inliers_dst))[0])) 
		y2 = np.asarray(list(list(zip(*inliers_dst))[1]))

		show_correspondence(images[i],images[i+1],x1,y1,x2,y2)

		if flag == 1:
			pass

		else:

			H = homography(inliers_src,inliers_dst)
			
			
			"""
			Image Warping + Blending
			Save Panorama output as mypano.png
			"""

			plt.figure()
			images[i+1] = combine(images[i+1],images[i],H)
			plt.imshow(images[i+1])
			plt.show()
			cv2.imwrite("mypano.png",images[i+1])
    
if __name__ == '__main__':
    main()
 
