#!/usr/bin/env python

"""
CMSC733 Spring 2020: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):

Chayan Kumar Patodi (ckp1804@terpmail.umd.edu)
University of Maryland, College Park

Saket Seshadri Gudimetla Hanumath (saketsgh@terpmail.umd.edu)
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import tensorflow as tf
import cv2
import sys
import os
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel_sup, HomographyModel_unsup 
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *
import copy

# Don't generate pyc codes
sys.dont_write_bytecode = True


def GenerateBatch(BasePath, DirNamesTrain, validation, MiniBatchSize, ModelType):
	"""
	Inputs:
	BasePath - Path to COCO folder without "/" at the end
	DirNamesTrain - Variable with Subfolder paths to train files
	NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
	TrainLabels - Labels corresponding to Train
	NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
	ImageSize - Size of the Image
	MiniBatchSize is the size of the MiniBatch
	Outputs:
	I1Batch - Batch of images
	LabelBatch - Batch of one-hot encoded labels
	ModelType - Supervised or Unsupervised Model
	validation - boolean variable to inform which dataset to load
	"""
	I1Batch = []
	LabelBatch = []

	ImageNum = 0

	if(ModelType == "sup"):
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
			for i in range(10):
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
				h4pt = np.subtract(patch_b, patch_a)
				h4pt = h4pt.reshape((8,))

				# restricting to [-1, 1]
				h4pt = h4pt/32

				# Append All Images and Mask
				I1Batch.append(input_data)
				LabelBatch.append(h4pt)

		return I1Batch, LabelBatch
	else:
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


			Img1_full.append(np.float32(im))
			Patches_.append(input_data)
			Corners_.append(np.float32(patch_a))
			I2_.append(np.float32(cropped_b.reshape(128,128,1)))

			# Label = convertToOneHot(TrainLabels[RandIdx], 10)
		return Img1_full, Patches_, Corners_, I2_


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
	"""
	Prints all stats with all arguments
	"""
	print('Number of Epochs Training will run for ' + str(NumEpochs))
	print('Factor of reduction in training data is ' + str(DivTrain))
	print('Mini Batch Size ' + str(MiniBatchSize))
	print('Number of Training Images ' + str(NumTrainSamples))
	if LatestFile is not None:
		print('Loading latest checkpoint with the name ' + LatestFile)


def TrainOperation_sup(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType):
	"""
	Inputs:
	ImgPH is the Input Image placeholder
	LabelPH is the one-hot encoded label placeholder
	DirNamesTrain - Variable with Subfolder paths to train files
	TrainLabels - Labels corresponding to Train/Test
	NumTrainSamples - length(Train)
	ImageSize - Size of the image
	NumEpochs - Number of passes through the Train data
	MiniBatchSize is the size of the MiniBatch
	SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
	CheckPointPath - Path to save checkpoints/model
	DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
	LatestFile - Latest checkpointfile to continue training
	BasePath - Path to COCO folder without "/" at the end
	LogsPath - Path to save Tensorboard Logs
	ModelType - Supervised or Unsupervised Model
	Outputs:
	Saves Trained network in CheckPointPath and Logs to LogsPath
	"""
	# creating placeholders for val image and labels
	# val_ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize*10, 128, 128, 2), name='val_ImgPH')
	# val_LabelPH = tf.placeholder(tf.float32, shape=(None, 8), name='val_LabelPH')


	# Predict output with forward pass
	H4Pt = HomographyModel_sup(ImgPH)

	with tf.name_scope('Loss'):
		###############################################
		# L2 loss
		###############################################
		train_loss = tf.sqrt(tf.reduce_sum((tf.squared_difference(H4Pt, LabelPH))))
		val_loss = tf.sqrt(tf.reduce_sum((tf.squared_difference(H4Pt, LabelPH))))

	with tf.name_scope('Adam'):
		###############################################
		# Adam Optimizer
		###############################################
		Optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(train_loss)

	# Tensorboard
	# Create a summary to monitor loss tensor
	train_loss_summary = tf.summary.scalar('TrainLossPerIter', train_loss)
	val_loss_summ = tf.summary.scalar('ValLossPerIter', val_loss)

	# define epoch loss ph
	train_epochloss_PH = tf.placeholder(tf.float32, shape=None, name='placeholder_epoch_train')
	train_epoch_loss_summary = tf.summary.scalar('TrainLossPerEpoch', train_epochloss_PH)

	# create placeholder for val loss
	val_epochloss_PH = tf.placeholder(tf.float32, shape=None, name='placeholder_epoch_val')
	val_epoch_loss_summary = tf.summary.scalar('ValLossPerEpoch', val_epochloss_PH)

	# Merge all summaries into a single operation
	MergedSummary_IterLoss_train = tf.summary.merge([train_loss_summary])
	MergedSummary_IterLoss_val = tf.summary.merge([val_loss_summ])

	MergedSummary_EpochLoss_train = tf.summary.merge([train_epoch_loss_summary])
	MergedSummary_EpochLoss_val = tf.summary.merge([val_epoch_loss_summary ])

	# Setup Saver
	Saver = tf.train.Saver()

	with tf.Session() as sess:
		if LatestFile is not None:
			Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
			# Extract only numbers from the name
			StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
			print('Loaded latest checkpoint with the name ' + LatestFile + '....')
		else:
			sess.run(tf.global_variables_initializer())
			StartEpoch = 0
			print('New model initialized....')

		# Tensorboard
		Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())

		for Epochs in tqdm(range(StartEpoch, NumEpochs)):

			NumIterationsPerEpoch = int(NumTrainSamples/(MiniBatchSize*10)/DivTrain)
			train_epoch_loss = 0
			val_epoch_loss = 0

			for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):

				# load training set and generate 100 images per iteration (500*100 = 50,000 sample images per epoch)
				I1Batch, LabelBatch = GenerateBatch(BasePath, DirNamesTrain, False, MiniBatchSize, ModelType)

				# load validation set and generate 100 images per iteration (500*100 = 50,000 sample images per epoch)
				val_I1Batch, val_LabelBatch = GenerateBatch(BasePath, DirNamesTrain, True, MiniBatchSize, ModelType)

				# obtain training loss
				FeedDict = {ImgPH: I1Batch, LabelPH: LabelBatch}
				_, trainLossThisBatch, IterLossSummary_train = sess.run([Optimizer, train_loss, MergedSummary_IterLoss_train], feed_dict=FeedDict)

				# obtain validation loss
				FeedDict = {ImgPH: val_I1Batch, LabelPH: val_LabelBatch}
				valLossThisBatch, IterLossSummary_val = sess.run([val_loss, MergedSummary_IterLoss_val], feed_dict=FeedDict)

				# calculate epoch losses
				train_epoch_loss += trainLossThisBatch
				val_epoch_loss += valLossThisBatch

				# print("***********************************\n{}".format(val_epoch_loss))

				# Save checkpoint every some SaveCheckPoint's iterations
				# if PerEpochCounter % SaveCheckPoint == 0:
				#     # Save the Model learnt in this epoch
				#     SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
				#     Saver.save(sess,  save_path=SaveName)
				#     print('\n' + SaveName + 'Model Saved...')

				# Tensorboard
				Writer.add_summary(IterLossSummary_train, Epochs*NumIterationsPerEpoch + PerEpochCounter)
				Writer.add_summary(IterLossSummary_val, Epochs*NumIterationsPerEpoch + PerEpochCounter)

			# Save model every epoch
			SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
			Saver.save(sess, save_path=SaveName)
			print('\n' + SaveName + ' Model Saved...')

			# average the epoch losses
			train_epoch_loss = train_epoch_loss/NumIterationsPerEpoch
			val_epoch_loss = val_epoch_loss/NumIterationsPerEpoch

			EpochLossSummary_train = sess.run(MergedSummary_EpochLoss_train, feed_dict={train_epochloss_PH: train_epoch_loss})
			EpochLossSummary_val = sess.run(MergedSummary_EpochLoss_val, feed_dict={val_epochloss_PH: val_epoch_loss})

			Writer.add_summary(EpochLossSummary_train, Epochs)
			Writer.add_summary(EpochLossSummary_val, Epochs)
			# If you don't flush the tensorboard doesn't update until a lot of iterations!
			Writer.flush()

def TrainOperation_unsup(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType):
	"""
	Inputs:
	ImgPH is the Input Image placeholder
	LabelPH is the one-hot encoded label placeholder
	DirNamesTrain - Variable with Subfolder paths to train files
	TrainLabels - Labels corresponding to Train/Test
	NumTrainSamples - length(Train)
	ImageSize - Size of the image
	NumEpochs - Number of passes through the Train data
	MiniBatchSize is the size of the MiniBatch
	SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
	CheckPointPath - Path to save checkpoints/model
	DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
	LatestFile - Latest checkpointfile to continue training
	BasePath - Path to COCO folder without "/" at the end
	LogsPath - Path to save Tensorboard Logs
	ModelType - Supervised or Unsupervised Model
	Outputs:
	Saves Trained network in CheckPointPath and Logs to LogsPath
	"""
	crnrPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 4, 2))
	I2_PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 128, 128, 1))

	I2_pred, I2 = HomographyModel_unsup(ImgPH, crnrPH, I2_PH, MiniBatchSize)


	with tf.name_scope('Loss'):
		###############################################
		# L2 loss
		###############################################
		train_loss = tf.reduce_mean(tf.abs(I2_pred - I2))
		val_loss = tf.reduce_mean(tf.abs(I2_pred - I2))

	with tf.name_scope('Adam'):
		###############################################
		# Adam Optimizer
		###############################################
		Optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(train_loss)

	# Tensorboard
	# Create a summary to monitor loss tensor
	train_loss_summary = tf.summary.scalar('TrainLossPerIter', train_loss)
	val_loss_summ = tf.summary.scalar('ValLossPerIter', val_loss)

	# define epoch loss ph
	train_epochloss_PH = tf.placeholder(tf.float32, shape=None, name='placeholder_epoch_train')
	train_epoch_loss_summary = tf.summary.scalar('TrainLossPerEpoch', train_epochloss_PH)

	# create placeholder for val loss
	val_epochloss_PH = tf.placeholder(tf.float32, shape=None, name='placeholder_epoch_val')
	val_epoch_loss_summary = tf.summary.scalar('ValLossPerEpoch', val_epochloss_PH)

	# Merge all summaries into a single operation
	MergedSummary_IterLoss_train = tf.summary.merge([train_loss_summary])
	MergedSummary_IterLoss_val = tf.summary.merge([val_loss_summ])

	MergedSummary_EpochLoss_train = tf.summary.merge([train_epoch_loss_summary])
	MergedSummary_EpochLoss_val = tf.summary.merge([val_epoch_loss_summary ])

	# Setup Saver
	Saver = tf.train.Saver()

	with tf.Session() as sess:
		if LatestFile is not None:
			Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
			# Extract only numbers from the name
			StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
			print('Loaded latest checkpoint with the name ' + LatestFile + '....')
		else:
			sess.run(tf.global_variables_initializer())
			StartEpoch = 0
			print('New model initialized....')

		# Tensorboard
		Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())

		for Epochs in tqdm(range(StartEpoch, NumEpochs)):

			NumIterationsPerEpoch = int(NumTrainSamples/(MiniBatchSize)/DivTrain)
			train_epoch_loss = 0
			val_epoch_loss = 0

			for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):

				# I1Batch, LabelBatch = GenerateBatch(BasePath, DirNamesTrain, False, MiniBatchSize, ModelType)
				Img1_full, Patches_, Corners_, I2_ = GenerateBatch(BasePath, DirNamesTrain, False, MiniBatchSize, ModelType)
				FeedDict = {ImgPH: Patches_, crnrPH: Corners_, I2_PH: I2_}

				_, trainLossThisBatch, IterLossSummary_train = sess.run([Optimizer, train_loss, MergedSummary_IterLoss_train], feed_dict=FeedDict)

				Img1_full, Patches_, Corners_, I2_ = GenerateBatch(BasePath, DirNamesTrain, True, MiniBatchSize, ModelType)
				FeedDict = {ImgPH: Patches_, crnrPH: Corners_, I2_PH: I2_}
				valLossThisBatch, IterLossSummary_val = sess.run([val_loss, MergedSummary_IterLoss_val], feed_dict=FeedDict)

				# calculate epoch losses
				train_epoch_loss += trainLossThisBatch
				val_epoch_loss += valLossThisBatch

				
				# Tensorboard
				Writer.add_summary(IterLossSummary_train, Epochs*NumIterationsPerEpoch + PerEpochCounter)
				Writer.add_summary(IterLossSummary_val, Epochs*NumIterationsPerEpoch + PerEpochCounter)

			# Save model every epoch
			SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
			Saver.save(sess, save_path=SaveName)
			print('\n' + SaveName + ' Model Saved...')

			# average the epoch losses
			train_epoch_loss = train_epoch_loss/NumIterationsPerEpoch
			val_epoch_loss = val_epoch_loss/NumIterationsPerEpoch

			EpochLossSummary_train = sess.run(MergedSummary_EpochLoss_train, feed_dict={train_epochloss_PH: train_epoch_loss})
			EpochLossSummary_val = sess.run(MergedSummary_EpochLoss_val, feed_dict={val_epochloss_PH: val_epoch_loss})

			Writer.add_summary(EpochLossSummary_train, Epochs)
			Writer.add_summary(EpochLossSummary_val, Epochs)
			# If you don't flush the tensorboard doesn't update until a lot of iterations!
			Writer.flush()



def main():
	"""
	Inputs:
	None
	Outputs:
	Runs the Training and testing code based on the Flag
	"""
	# Parse Command Line arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--BasePath', default='../Data/', help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/COCO')
	Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
	Parser.add_argument('--ModelType', default='sup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
	Parser.add_argument('--NumEpochs', type=int, default=111, help='Number of Epochs to Train for, Default:50')
	Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
	Parser.add_argument('--MiniBatchSize', type=int, default=10, help='Size of the MiniBatch to use, Default:1')
	Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
	Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')

	Args = Parser.parse_args()
	NumEpochs = Args.NumEpochs
	BasePath = Args.BasePath
	DivTrain = float(Args.DivTrain)
	MiniBatchSize = Args.MiniBatchSize
	LoadCheckPoint = Args.LoadCheckPoint
	CheckPointPath = Args.CheckPointPath
	LogsPath = Args.LogsPath
	ModelType = Args.ModelType
	ModelType = ModelType.lower()

	# Setup all needed parameters including file reading
	# DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)
	DirNamesTrain, SaveCheckPoint, ImageSize = SetupAll(BasePath, CheckPointPath)

	TrainLabels = None
	NumClasses = None

	patch_width = 128
	patch_height = 128

	# Define PlaceHolder variables for Input and Predicted output
	LabelPH = tf.placeholder(tf.float32, shape=(None, 8), name='LabelPH') # labels
	ImgPH = tf.placeholder(tf.float32, shape=(None, patch_height, patch_width, 2), name='ImgPH')

	if(ModelType == "sup"):
		if LoadCheckPoint==1:
			LatestFile = FindLatestModel(CheckPointPath)
		else:
			LatestFile = None

		# ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize*10, patch_height, patch_width, 2))
		NumTrainSamples = 50000
		PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

		TrainOperation_sup(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,DivTrain,
				   LatestFile, BasePath, LogsPath, ModelType)

	else :
		CheckPointPath+="unsup/"
		LogsPath+="unsup/"
		if LoadCheckPoint==1:
			LatestFile = FindLatestModel(CheckPointPath)
		else:
			LatestFile = None

		MiniBatchSize = 32
		NumTrainSamples = 5000	
		PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)
		TrainOperation_unsup(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,DivTrain,
				   LatestFile, BasePath, LogsPath, ModelType)


if __name__ == '__main__':
	main()
 
