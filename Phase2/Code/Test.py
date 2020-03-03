#!/usr/bin/env python

"""
CMSC733 Spring 2020: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


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

import tensorflow as tf
import cv2
import os
import sys
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel_sup
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *
import copy

# Don't generate pyc codes
sys.dont_write_bytecode = True

# def SetupAll(BasePath):
#     """
#     Inputs: 
#     BasePath - Path to images
#     Outputs:
#     ImageSize - Size of the Image
#     DataPath - Paths of all images where testing will be run on
#     """   
#     # Image Input Shape
#     ImageSize = [32, 32, 3]
#     DataPath = []
#     NumImages = len(glob.glob(BasePath+'*.jpg'))
#     SkipFactor = 1
#     for count in range(1,NumImages+1,SkipFactor):
#         DataPath.append(BasePath + str(count) + '.jpg')

#     return ImageSize, DataPath
    
def ReadImages(DataPath):
    """
    Inputs: 
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    
    RandIdx = random.randrange(1, 1000)

    rand_image = cv2.imread(DataPath+str(RandIdx)+".jpg")

    # patch size
    patch_width = 128
    patch_height = 128

    # perturbation factor p 
    p = 32

    # copy of img
    im = copy.deepcopy(rand_image)

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
    # h4pt = np.subtract(patch_b, patch_a)
    # h4pt = h4pt.reshape((8,))
    # h4pt = h4pt/p

    return patch_a, patch_b, input_data, rand_image
    
    # I1 = cv2.imread(ImageName)
    
    # if(I1 is None):
    #     # OpenCV returns empty list if image is not read! 
    #     print('ERROR: Image I1 cannot be read')
    #     sys.exit()
        
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################

    # I1S = iu.StandardizeInputs(np.float32(I1))

    # I1Combined = np.expand_dims(I1S, axis=0)

    # return I1Combined, I1
                

def TestOperation(ImgPH, ImageSize, ModelPath, DataPath):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    # """
    # Length = ImageSize[0]
    # # Predict output with forward pass, MiniBatchSize for Test is 1
    # _, prSoftMaxS = CIFAR10Model(ImgPH, ImageSize, 1)

    H4Pt = HomographyModel_sup(ImgPH)
    patch_a, patch_b, input_data, img = ReadImages(DataPath)

    # Setup Saver
    Saver = tf.train.Saver()

    
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        
        input_data = np.array(input_data).reshape(1, input_data.shape[0], input_data.shape[1], input_data.shape[2])
        FeedDict = {ImgPH:input_data}
        predict_perturb = sess.run(H4Pt, FeedDict)
        predict_perturb = predict_perturb.reshape((4, 2))
        predict_perturb = 32*predict_perturb
        # OutSaveT = open(LabelsPathPred, 'w')
        # for count in tqdm(range(np.size(DataPath))):            
        #     DataPathNow = DataPath[count]
        #     Img, ImgOrg = ReadImages(ImageSize, DataPathNow)
        #     FeedDict = {ImgPH: Img}
        #     PredT = np.argmax(sess.run(prSoftMaxS, FeedDict))

        #     OutSaveT.write(str(PredT)+'\n')
            
        # OutSaveT.close()

    patch_c = patch_a + predict_perturb

    # draw all of the patches for comparison
    plot_a = np.array(patch_a, np.int32)
    plot_a = plot_a.reshape((-1, 1, 2))
    
    plot_b = np.array(patch_b, np.int32)
    plot_b = plot_b.reshape((-1, 1, 2))
    
    plot_c = np.array(patch_c, np.int32)
    plot_c = plot_c.reshape((-1, 1, 2))

    # cv2.polylines(img, [plot_a] , True, (255,0,0), 5)
    cv2.polylines(img, [plot_b], True, (0,255,0), 5)
    cv2.polylines(img, [plot_c], True, (0,0,255), 5)
    cv2.imwrite("Final_Output.png", img)
    # cv2.imshow('Final_Output', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# def ReadLabels(LabelsPathTest, LabelsPathPred):
#     if(not (os.path.isfile(LabelsPathTest))):
#         print('ERROR: Test Labels do not exist in '+LabelsPathTest)
#         sys.exit()
#     else:
#         LabelTest = open(LabelsPathTest, 'r')
#         LabelTest = LabelTest.read()
#         LabelTest = map(float, LabelTest.split())

#     if(not (os.path.isfile(LabelsPathPred))):
#         print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
#         sys.exit()
#     else:
#         LabelPred = open(LabelsPathPred, 'r')
#         LabelPred = LabelPred.read()
#         LabelPred = map(float, LabelPred.split())
        
#     return LabelTest, LabelPred

        
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelType', default='sup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Parser.add_argument('--ModelPath', dest='ModelPath', default='../Checkpoints/110model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='../Data/Test/Phase2/', help='Path to load images from, Default:BasePath')

    Args = Parser.parse_args()
    ModelType = Args.ModelType
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    # ModelType = ModelType.lower()

    # Setup all needed parameters including file reading
    # ImageSize, DataPath = SetupAll(BasePath)
    DataPath = BasePath
    LabelsPathPred = "empty"

    ImageSize = (1, 128, 128, 2)

    # if(ModelType=="unsup"):
    #     ModelPath = "../Checkpoints/unsup/21model.ckpt"

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder('float', shape=(ImageSize[0], ImageSize[1], ImageSize[2], ImageSize[3]))
    # LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

    TestOperation(ImgPH, ImageSize, ModelPath, DataPath)

    # Plot Confusion Matrix
    # LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    # ConfusionMatrix(LabelsTrue, LabelsPred)
     
if __name__ == '__main__':
    main()
 
