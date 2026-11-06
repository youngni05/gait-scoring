#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:53:26 2019

@author: 55129822
"""
'''
    consider tracked?
'''

import numpy as np
import pandas as pd
import os

from scipy import signal
from scipy import stats
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.image as mpl_image

from PIL import Image as pil_image

from mpl_toolkits.mplot3d import Axes3D

import os
import sys
from tqdm import tqdm

from enum import Enum
import re

from sway_utils import metrics as sm

import csv

from sklearn import preprocessing
import scipy.misc
#%%
class SkeletonJoints(Enum):
    """
    This is an enumeration class that defines the different joints in a skeleton. 
    Each joint is assigned a unique integer value.
    """
    SPINEBASE = 0
    SPINEMID = 1
    NECK = 2
    HEAD = 3
    SHOULDERLEFT = 4
    ELBOWLEFT = 5
    WRISTLEFT = 6
    HANDLEFT = 7
    SHOULDERRIGHT = 8
    ELBOWRIGHT = 9
    WRISTRIGHT = 10
    HANDRIGHT = 11
    HIPLEFT = 12
    KNEELEFT = 13
    ANKLELEFT = 14
    FOOTLEFT = 15
    HIPRIGHT = 16
    KNEERIGHT = 17
    ANKLERIGHT = 18
    FOOTRIGHT = 19
    SPINESHOULDER = 20
    HANDTIPLEFT = 21
    THUMBLEFT = 22
    HANDTIPRIGHT = 23
    THUMBRIGHT = 24
    COM = 25
    #HEELLEFT = 26
    #HEELRIGHT = 27
    
    
class SkeletonJoints_no_hands_or_feet(Enum):
    """
    Same as SkeletonJoints enumeration but exludes hands and feet.
    Usfull if the hand recordings are partularlly noisey.
    """
    SPINEBASE = 0
    SPINEMID = 1
    NECK = 2
    #HEAD = 3
    SHOULDERLEFT = 4
    ELBOWLEFT = 5
    WRISTLEFT = 6
    #HANDLEFT = 7
    SHOULDERRIGHT = 8
    ELBOWRIGHT = 9
    WRISTRIGHT = 10
    #HANDRIGHT = 11
    HIPLEFT = 12
    KNEELEFT = 13
    ANKLELEFT = 14
    #FOOTLEFT = 15
    HIPRIGHT = 16
    KNEERIGHT = 17
    ANKLERIGHT = 18
    #FOOTRIGHT = 19
    SPINESHOULDER = 20
    #HANDTIPLEFT = 21
    #THUMBLEFT = 22
    #HANDTIPRIGHT = 23
    #THUMBRIGHT = 24
    COM = 25
    #HEELLEFT = 26
    #HEELRIGHT = 27
    
    
    
class HierarchicalSkeletonJoints(Enum):
    """
    This is an enumeration class that defines the joints of a hierarchical skeleton. 
    Each joint is assigned a unique integer value.
    """
    COM = 25
    
    HEAD = 3
    NECK = 2
    SPINESHOULDER = 20
    SPINEMID = 1
    SPINEBASE = 0
    
    SHOULDERLEFT = 4
    SHOULDERRIGHT = 8
    
    ELBOWLEFT = 5
    ELBOWRIGHT = 9
    
    WRISTLEFT = 6
    WRISTRIGHT = 10
    
    HIPLEFT = 12
    HIPRIGHT = 16
    
    KNEELEFT = 13
    KNEERIGHT = 17
    
    ANKLELEFT = 14
    ANKLERIGHT = 18
    
    FOOTLEFT = 15
    FOOTRIGHT = 19
    
    HANDLEFT = 7
    HANDTIPLEFT = 21
    THUMBLEFT = 22
    
    HANDRIGHT = 11
    HANDTIPRIGHT = 23
    THUMBRIGHT = 24
    
    
class SkeletonJointAngles(Enum):
    """
    This is an enumeration class that defines a set of named constants representing the different types of joint angles in a skeleton. 
    Each constant is assigned a unique integer value.
    """
    BODY_COM_ANGLE = 1
    BODY_LEAN_ANGLE = 2
    KNEELEFT_ANGLE = 3
    KNEERIGHT_ANGLE = 4
    HIPLEFT_ANGLE = 5
    HIPRIGHT_ANGLE = 6
    ELBOWLEFT_ANGLE = 7
    ELBOWLRIGHT_ANGLE = 8
    ARMPITLEFT_ANGLE = 9
    ARMPITROIGHT_ANGLE = 10
    ANKLELEFT_ANGLE = 11
    ANKLELRIGHT_ANGLE = 12
    
    
class BodySegments(Enum):
    """
    This is an enumeration class that defines the different body segments of a skeleton. 
    Each body segment is defined as a list of two skeleton joints.
    """
    #Torso
    HEAD_NECK = [SkeletonJoints.HEAD.value, SkeletonJoints.NECK.value]
    NECK_SPINESHOULDER = [SkeletonJoints.NECK.value, SkeletonJoints.SPINESHOULDER.value]
    SPINESHOULDER_SPINEMID = [SkeletonJoints.SPINESHOULDER.value, SkeletonJoints.SPINEMID.value]
    SPINEMID_SPINEBASE = [SkeletonJoints.SPINEMID.value, SkeletonJoints.SPINEBASE.value] 
    
    #Left arm
    SPINESHOULDER_SHOULDERLEFT = [SkeletonJoints.SPINESHOULDER.value, SkeletonJoints.SHOULDERLEFT.value]
    SHOULDERLEFT_ELBOWLEFT = [SkeletonJoints.SHOULDERLEFT.value, SkeletonJoints.ELBOWLEFT.value]
    ELBOWLEFT_WRISTLEFT = [SkeletonJoints.ELBOWLEFT.value, SkeletonJoints.WRISTLEFT.value]
    WRISTLEFT_HANDLEFT = [SkeletonJoints.WRISTLEFT.value, SkeletonJoints.HANDLEFT.value]
    HANDLEFT_THUMPLEFT = [SkeletonJoints.HANDLEFT.value, SkeletonJoints.THUMBLEFT.value]
    HANDLEFT_HANDTIPLEFT = [SkeletonJoints.HANDLEFT.value, SkeletonJoints.HANDTIPLEFT.value]
    
    #Right arm
    SPINESHOULDER_SHOULDERRIGHT = [SkeletonJoints.SPINESHOULDER.value, SkeletonJoints.SHOULDERRIGHT.value]
    SHOULDERRIGHT_ELBOWRIGHT = [SkeletonJoints.SHOULDERRIGHT.value, SkeletonJoints.ELBOWRIGHT.value]
    ELBOWRIGHT_WRISTRIGHT = [SkeletonJoints.ELBOWRIGHT.value, SkeletonJoints.WRISTRIGHT.value]
    WRISTRIGHT_HANDRIGHT = [SkeletonJoints.WRISTRIGHT.value, SkeletonJoints.HANDRIGHT.value]
    HANDRIGHT_THUMPRIGHT = [SkeletonJoints.HANDRIGHT.value, SkeletonJoints.THUMBRIGHT.value]
    HANDRIGHT_HANDTIPRIGHT = [SkeletonJoints.HANDRIGHT.value, SkeletonJoints.HANDTIPRIGHT.value]
    
    #Left leg
    SPINEBASE_HIPLEFT = [SkeletonJoints.SPINEBASE.value, SkeletonJoints.HIPLEFT.value]
    HIPLEFT_KNEELEFT = [SkeletonJoints.HIPLEFT.value, SkeletonJoints.KNEELEFT.value]
    KNEELEFT_ANKLELEFT = [SkeletonJoints.KNEELEFT.value, SkeletonJoints.ANKLELEFT.value]
    ANKLELEFT_FOOTLEFT = [SkeletonJoints.ANKLELEFT.value, SkeletonJoints.FOOTLEFT.value]
    
    #Right leg
    SPINEBASE_HIPRIGHT = [SkeletonJoints.SPINEBASE.value, SkeletonJoints.HIPRIGHT.value]
    HIPRIGHT_KNEERIGHT = [SkeletonJoints.HIPRIGHT.value, SkeletonJoints.KNEERIGHT.value]
    KNEERIGHT_ANKLERIGHT = [SkeletonJoints.KNEERIGHT.value, SkeletonJoints.ANKLERIGHT.value]
    ANKLERIGHT_FOOTRIGHT = [SkeletonJoints.ANKLERIGHT.value, SkeletonJoints.FOOTRIGHT.value]

    HEAD_SPINE_BASE = [SkeletonJoints.HEAD.value, SkeletonJoints.SPINEBASE.value]
    SHOULDERKEFT_WRISTLEFT = [SkeletonJoints.SHOULDERLEFT.value, SkeletonJoints.WRISTLEFT.value]
    HIPLEFT_ANKLELEFT = [SkeletonJoints.HIPLEFT.value, SkeletonJoints.ANKLELEFT.value]


class HierarchyForBodySegments(Enum):
    """
    This is an enumeration class that defines the hierarchy of body segments. 
    Each body segment is assigned a unique integer value.
    """
    HEAD = 3
    NECK = 2
    SPINESHOULDER = 20
    SPINEMID = 1
    SPINEBASE = 0
    
    SHOULDERLEFT = 4
    SHOULDERRIGHT = 8
    
    ELBOWLEFT = 5
    ELBOWRIGHT = 9
    
    WRISTLEFT = 6
    WRISTRIGHT = 10
    
    HIPLEFT = 12
    HIPRIGHT = 16
    
    KNEELEFT = 13
    KNEERIGHT = 17
    
    ANKLELEFT = 14
    ANKLERIGHT = 18
    
    FOOTLEFT = 15
    FOOTRIGHT = 19
    
    HANDLEFT = 7
    HANDTIPLEFT = 21
    THUMBLEFT = 22
    
    HANDRIGHT = 11
    HANDTIPRIGHT = 23
    THUMBRIGHT = 24


class BodyParts(Enum):
    """
    This is an enumeration class that defines the different body parts and their corresponding skeleton joints collections
    """
    TORSO = [SkeletonJoints.HEAD,
             SkeletonJoints.NECK,
             SkeletonJoints.SPINESHOULDER,
             SkeletonJoints.SPINEMID,
             SkeletonJoints.SPINEBASE]
    
    ARM_LEFT = [SkeletonJoints.SHOULDERLEFT, 
                SkeletonJoints.ELBOWLEFT,
                SkeletonJoints.WRISTLEFT,
                SkeletonJoints.HANDLEFT,
                SkeletonJoints.THUMBLEFT,
                SkeletonJoints.HANDTIPLEFT]
    
    ARM_RIGHT = [SkeletonJoints.SHOULDERRIGHT, 
                 SkeletonJoints.ELBOWRIGHT,
                 SkeletonJoints.WRISTRIGHT,
                 SkeletonJoints.HANDRIGHT,
                 SkeletonJoints.THUMBRIGHT,
                 SkeletonJoints.HANDTIPRIGHT]
    
    LEG_LEFT = [SkeletonJoints.HIPLEFT,
                SkeletonJoints.KNEELEFT,
                SkeletonJoints.ANKLELEFT,
                SkeletonJoints.FOOTLEFT]
    
    LEG_RIGHT = [SkeletonJoints.HIPRIGHT,
                 SkeletonJoints.KNEERIGHT,
                 SkeletonJoints.ANKLERIGHT,
                 SkeletonJoints.FOOTRIGHT]


class BodySections(Enum):
    """
    This is an enumeration class that defines three body sections: torso, arms, and legs. 
    Each body section is defined as a list of body parts. 
    """
    TORSO = BodyParts.TORSO
    ARMS = [BodyParts.ARM_LEFT, BodyParts.ARM_RIGHT]
    LEGS = [BodyParts.LEG_LEFT, BodyParts.LEG_RIGHT]
    
    
class ScaleBodyParts(Enum):
    """
    This is an enumeration class that defines three different body part scales: torso, arms, and legs. 
    Each scale is defined as a list of body part names.
    """
    TORSO = ['HEAD',
            'NECK',
            'SPINESHOULDER',
            'SPINEMID',
            'SPINEBASE']
    
    ARMS = ['SHOULDERLEFT', 
           'EBOWLEFT',
           'WRISTLEFT',
           'HANDLEFT',
           'THUMBLEFT',
           'HANDTIPLEFT',
           'SHOULDERRIGHT', 
           'ELBOWRIGHT',
           'WRISTRIGHT',
           'HANDRIGHT',
           'THUMBRIGHT',
           'HANDTIPRIGHT']
    
    LEGS = ['HIPLEFT',
           'KNEELEFT',
           'ANKLELEFT',
           'FOOTLEFT',
           'HIPRIGHT',
           'KNEERIGHT',
           'ANKLERIGHT',
           'FOOTRIGHT']
   

class WalkedSkelAngles(Enum):
    """
    This is an enumeration class that defines the different angles between joints in a human skeleton. 
    
    inspired by: 
    A. Vakanski, H. P. Jun, D. Paul, and R. Baker,
    “A data set of human body movements for physical rehabilitation exercises,”
    Data, vol. 3, no. 1, 2018.
    NB no head tip and no tip
    """
    #Torso
    SPINEBASE_SPINEMID = [SkeletonJoints.SPINEBASE.value, SkeletonJoints.SPINEMID.value]
    SPINEMID_SPINESHOULDER = [SkeletonJoints.SPINEMID.value, SkeletonJoints.SPINESHOULDER.value]
    SPINESHOULDER_NECK = [SkeletonJoints.SPINESHOULDER.value, SkeletonJoints.NECK.value]
    NECK_HEAD = [SkeletonJoints.NECK.value, SkeletonJoints.HEAD.value]
    
    #Left upper
    SPINESHOULDER_SHOULDERLEFT = [SkeletonJoints.SPINESHOULDER.value, SkeletonJoints.SHOULDERLEFT.value]
    SHOULDERLEFT_ELBOWLEFT = [SkeletonJoints.SHOULDERLEFT.value, SkeletonJoints.ELBOWLEFT.value]
    ELBOWLEFT_WRISTLEFT = [SkeletonJoints.ELBOWLEFT.value, SkeletonJoints.WRISTLEFT.value]
    WRISTLEFT_HANDLEFT = [SkeletonJoints.WRISTLEFT.value, SkeletonJoints.HANDLEFT.value]

    #Right upper
    SPINESHOULDER_SHOULDERRIGHT = [SkeletonJoints.SPINESHOULDER.value, SkeletonJoints.SHOULDERRIGHT.value]
    SHOULDERRIGHT_ELBOWRIGHT = [SkeletonJoints.SHOULDERRIGHT.value, SkeletonJoints.ELBOWRIGHT.value]
    ELBOWRIGHT_WRISTRIGHT = [SkeletonJoints.ELBOWRIGHT.value, SkeletonJoints.WRISTRIGHT.value]
    WRISTRIGHT_HANDRIGHT = [SkeletonJoints.WRISTRIGHT.value, SkeletonJoints.HANDRIGHT.value]
    
    #Left lower
    SPINEBASE_HIPLEFT = [SkeletonJoints.SPINEBASE.value, SkeletonJoints.HIPLEFT.value]
    HIPLEFT_KNEELEFT = [SkeletonJoints.HIPLEFT.value, SkeletonJoints.KNEELEFT.value]
    KNEELEFT_ANKLELEFT = [SkeletonJoints.KNEELEFT.value, SkeletonJoints.ANKLELEFT.value]
    ANKLELEFT_FOOTLEFT = [SkeletonJoints.ANKLELEFT.value, SkeletonJoints.FOOTLEFT.value]
    
    #Right lower
    SPINEBASE_HIPRIGHT = [SkeletonJoints.SPINEBASE.value, SkeletonJoints.HIPRIGHT.value]
    HIPRIGHT_KNEERIGHT = [SkeletonJoints.HIPRIGHT.value, SkeletonJoints.KNEERIGHT.value]
    KNEERIGHT_ANKLERIGHT = [SkeletonJoints.KNEERIGHT.value, SkeletonJoints.ANKLERIGHT.value]
    ANKLERIGHT_FOOTRIGHT = [SkeletonJoints.ANKLERIGHT.value, SkeletonJoints.FOOTRIGHT.value]
    
    
class WalkedSkelAnglesIn3s(Enum):
    """
    This is an enumeration class that defines the angles between various joints in a human skeleton. 
    Each angle is defined as a list of three joints, with the angle being the angle between the second joint and the line connecting the first and third joints. 
    """
    #Torso
    SPINEMID_a = [SkeletonJoints.SPINEBASE.value, 
                  SkeletonJoints.SPINEMID.value,
                  SkeletonJoints.SPINESHOULDER.value]
    SPINESHOULDER_a = [SkeletonJoints.SPINEMID.value,
                       SkeletonJoints.SPINESHOULDER.value,
                       SkeletonJoints.NECK.value]
    NECK_a = [SkeletonJoints.SPINESHOULDER.value,
              SkeletonJoints.NECK.value,
              SkeletonJoints.HEAD.value]
    
    #Upper body
    ARMPITLEFT_a = [SkeletonJoints.SPINESHOULDER.value,
                    SkeletonJoints.SHOULDERLEFT.value,
                    SkeletonJoints.ELBOWLEFT.value]
    ELBOWLEFT_a = [SkeletonJoints.SHOULDERLEFT.value,
                   SkeletonJoints.ELBOWLEFT.value,
                   SkeletonJoints.WRISTLEFT.value]
    
    ARMPITRIGHT_a = [SkeletonJoints.SPINESHOULDER.value,
                     SkeletonJoints.SHOULDERRIGHT.value,
                     SkeletonJoints.ELBOWRIGHT.value]
    ELBOWRIGHT_a = [SkeletonJoints.SHOULDERLEFT.value,
                    SkeletonJoints.ELBOWRIGHT.value,
                    SkeletonJoints.WRISTRIGHT.value]

    #Lower body
    HIPLEFT_a = [SkeletonJoints.SPINEBASE.value,
                 SkeletonJoints.HIPLEFT.value,
                 SkeletonJoints.KNEELEFT.value]
    KNEELEFT_a = [SkeletonJoints.HIPLEFT.value,
                  SkeletonJoints.KNEELEFT.value,
                  SkeletonJoints.ANKLELEFT.value]
    
    HIPRIGHT_a = [SkeletonJoints.SPINEBASE.value,
                  SkeletonJoints.HIPRIGHT.value,
                  SkeletonJoints.KNEERIGHT.value]
    KNEERIGHT_a = [SkeletonJoints.HIPLEFT.value,
                   SkeletonJoints.KNEERIGHT.value,
                   SkeletonJoints.ANKLERIGHT.value]
    
class RefernceTorsoJoints_HEAD(Enum):
    """
    This is a class that defines an enumeration of reference torso joints. 
    """
    HEAD = SkeletonJoints.HEAD.value
   
    SHOULDERLEFT = SkeletonJoints.SHOULDERLEFT.value
    SHOULDERRIGHT = SkeletonJoints.SHOULDERRIGHT.value    
    
    HIPLEFT = SkeletonJoints.HIPLEFT.value
    HIPRIGHT = SkeletonJoints.HIPRIGHT.value


class RefernceTorsoJoints_NECK(Enum):
    """
    This is a class that defines an enumeration of reference torso joints.
    """
    NECK = SkeletonJoints.NECK.value
    
    SHOULDERLEFT = SkeletonJoints.SHOULDERLEFT.value
    SHOULDERRIGHT = SkeletonJoints.SHOULDERRIGHT.value
        
    HIPLEFT = SkeletonJoints.HIPLEFT.value
    HIPRIGHT = SkeletonJoints.HIPRIGHT.value 


class RefernceTorsoJoints_COM(Enum):
    """
    This is a class enumerateds a single COM joint
    """
    COM = SkeletonJoints.COM.value
       

"""
This associates X,Y,X with 0, 1, and 2.
This makes working with 3D arrays that represent the movement easier.
"""
X = 0
Y = 1
Z = 2

#%%
'''
General utilities
'''

def walkFileSystem(filePath):
    """
    Given a file path, walk through the file system and return the root, directories, and files and removes extraneous folders from the list

    @param filePath - the file path to walk through
    @return the root, directories, and files
    """
    root = []
    dirs = []
    files = []

    for root, dirs, files in os.walk(filePath):
        break

    d = ''
    for d in dirs:
        if 'remov' in d.lower():
            dirs.remove(d)

    d = ''
    for d in dirs:
        if 'up-and-go' in d.lower():
            dirs.remove(d)

    d = ''
    for d in dirs:
        if '3m' in d.lower():
            dirs.remove(d)

    d = ''
    for d in dirs:
        if 'toe' in d.lower():
            dirs.remove(d)

    d = ''
    for d in dirs:
        if 'dual' in d.lower():
            dirs.remove(d)


    return root, dirs, files

#%%
class KinectRecording:
    """
    This is a class that represents a single kinect recording. 
    """
    _root_path = ''
    _skel_root_path = '' 
    _skel_file_root = ''
    
    _dataset_prefix = ''
    _movement = ''
    _part_id = 0
    _labels = []
    _ref_spine_base = [] #spine base of first skel frame
    _ref_skel_frame = [] #whole      of first skel frame
    
    _frame_count = -1
    _load_from_cache = False

    _cached_stacked_raw_XYZ_file_path = ''
    _cached_skeletons_path = ''
    
    _skel_scale = []
    _torso_scale = 0
    #_ref_neck_length = 0.15
    _scale_skeletons = False

    ''''
        Values from files
        skeletons: list of pandas DataFrames, representing 
        text files from kinect: 
            [features, SkeletonJoints] tipically, [6,26]
            fetures : Indes, Tracked, X, Y, Z, px_X, px_Y
    '''
    skeletons = []
    raw_XYZ_values = []
    
    ''''
        stacked XYZ values are flattend versions of the raw and filteterd 
        kinect recordings
        [XYZ, SkeletonJoints, #frames] typically,[3, 26, 600]
    '''
    stacked_raw_XYZ_values = []
    stacked_filtered_XYZ_values = []
    
    ''''
        stacked angle values represent the key angles of the 
        filtered XYZ values 
        [fetures, #frames] typically,[3, 600]
        featurs: COM_angle, knee_angle, hip_angle_side, hip_angle_front,
                lean_angle
    '''
    
    stacked_raw_angle_values = []
    stacked_filtered_angle_vlaues = []
    
    
    ''' 
        get sway metrics from file
    '''
    sway_metrics = []

    
    '''
        calulate and store    
        local inter-joint coordination pattern (IJCP)
    ''' 
    IJCP_Dist = []
    IJCP_Velo = []
    
    
    '''
        Calulate and store smc features
    '''
    
    smc_features = []
    
    '''
        calulate and store Walling Skel Angles
    '''
    walked_skel_angles = []
    
    
    '''
        Calulate and sotre Cosine Distance and Normalised Magnitude from 
        Q. Ke, S. An, M. Bennamoun, F. Sohel, and F. Boussaid, “SkeletonNet: 
            Mining Deep Part Features for 3-D Action Recognition,” 
            IEEE Signal Process. Lett., vol. 24, no. 6, pp. 731–735, Jun. 2017.
    '''
    cosine_distance = []
    normalised_magnitude = []
    
    
    
    def __init__(self, skel_root_path, dataset_prefix, movement, part_id, labels=[], scale_skeletons=False):
        """
        This is the constructor for a class that loads skeleton data and sway metrics.

        @param skel_root_path - the path to the skeleton data
        @param dataset_prefix - the prefix for the dataset
        @param movement - the movement being analyzed
        @param part_id - the part ID
        @param labels - the labels for the data
        @param scale_skeletons - whether or not to scale the skeletons
        """
        self._root_path = str.replace(skel_root_path, '/skel', '')
        self._root_path = str.replace(self._root_path, '\skel', '')
        self._dataset_prefix = dataset_prefix
        self._skel_root_path = skel_root_path
        self._movement = movement
        self._part_id = part_id
        self._labels = labels
        self._scale_skeletons = scale_skeletons
        
        self.load_skeletons(skel_root_path)
        self.load_sway_metrics()
        
        
    def load_sway_metrics(self):
        """
        Load sway metrics from a CSV file located in the root path of the object.
        If the file exists, read the CSV file and store the data in the object's sway_metrics attribute.
        """
        sway_metric_path = os.path.join(self._root_path, 'sway_metrics.csv')
        if os.path.exists(sway_metric_path):
            self.sway_metrics = pd.read_csv(sway_metric_path)
        
        
    def save_stacked_raw_XYZ(self):
        """
        Caches the stacked raw XYZ values to a file if the file does not already exist.
        """
        if not os.path.exists(self._cached_stacked_raw_XYZ_file_path):
            np.save(self._cached_stacked_raw_XYZ_file_path, self.stacked_raw_XYZ_values)
            
            
    def save_skeletons(self):
        """
        Save the skeletons to a CSV file if the file does not already exist.
        """
        if not os.path.exists(self._cached_skeletons_path):
            #np.save(self._cached_skeletons_path, self.skeletons)
            pd.concat(self.skeletons).to_csv(self._cached_skeletons_path)
        
    
    def load_skeletons(self, skel_root_path):
        #If chahed file exists, load
        """
        Loads, normalised raw skeleton files then caches as processed files to save time next time around.

        @param skel_root_path - the path to the skeleton data
        """
        
        #Load scaled or none-scaled versions
        if self._scale_skeletons:
            cached_stacked_raw_XYZ_file_name = (self._dataset_prefix +
                                                str(self._part_id) + '_' +
                                                'cached_stacked_raw_XYZ_scaled_' +
                                                #'cached_stacked_ankle_raw_XYZ_scaled_' +
                                                self._movement + '.npy')
            
            cached_skeletons_file_name = (self._dataset_prefix +
                                                str(self._part_id) + '_' +
                                                'cached_skeletons_scaled_' +
                                                #'cached_skeletons_ankle_scaled_' + 
                                                self._movement + '.csv') 
        
        else:
            cached_stacked_raw_XYZ_file_name = (self._dataset_prefix +
                                                str(self._part_id) + '_' +
                                                'cached_stacked_raw_XYZ_' + 
                                                self._movement + '.npy')
            
            cached_skeletons_file_name = (self._dataset_prefix +
                                                str(self._part_id) + '_' +
                                                'cached_skeletons' + 
                                                self._movement + '.csv') 
                                            
        self._cached_stacked_raw_XYZ_file_path = os.path.join(self._root_path, 
                                                        cached_stacked_raw_XYZ_file_name) 
        
        self._cached_skeletons_path = os.path.join(self._root_path, 
                                                        cached_skeletons_file_name) 
        
        if os.path.exists(self._cached_stacked_raw_XYZ_file_path):
            self.stacked_raw_XYZ_values = np.load(self._cached_stacked_raw_XYZ_file_path)
            self._load_from_cache = True
            #print('loading:', self._cached_stacked_raw_XYZ_file_path, '\n')
            
            
        if os.path.exists(self._cached_skeletons_path):
            self.skeletons = pd.read_csv(self._cached_skeletons_path)
            #self._load_from_cache = True
            #print('loading:', self._cached_skeletons_path, '\n')    
        
        #else calulate stuff
        else:
            root, dirs, skel_files = walkFileSystem(skel_root_path)
            skel_files.sort()
            
            if len(skel_files) == 0:
                print('Skel files for:', self._cached_stacked_raw_XYZ_file_path)
         
            for skelfile in tqdm(skel_files):
            #for skelfile in skel_files:    
                skel_file_path = os.path.join(root, skelfile)
                _skel_frame, _raw_XYZ = self._load_skel_file(skel_file_path)
                '''skels are now normailesed and com added '''
                
                self.skeletons.append(_skel_frame)
                self.raw_XYZ_values.append(_raw_XYZ)
                
                #dont include first frame, having a skel base of 0,0,0 can cause problems
                if self._frame_count > 0:
                    if len(self.stacked_raw_XYZ_values) == 0:
                        self.stacked_raw_XYZ_values = _raw_XYZ
                    else:
                        self.stacked_raw_XYZ_values = np.dstack([self.stacked_raw_XYZ_values, _raw_XYZ])
        
        #Save
        self.save_stacked_raw_XYZ()
        
        self.save_skeletons()
        
        #now calulate features
        self.stacked_filtered_XYZ_values = self.filter_joint_sequences(self.stacked_raw_XYZ_values)
    
        return
        
        
    def filter_joint_sequences(self, noisy_raw_XYZ, N=2, fc=10, fs=30):        
        """
        This function filters joint sequences using the flilter singal funtion form metrics.py

        @param self - the class instance
        @param noisy_raw_XYZ - the noisy joint sequences
        @param N - the order of the filter
        @param fc - the cutoff frequency of the filter
        @param fs - the sampling frequency of the signal
        @return the filtered joint sequences
        """
        stacked_filtered_XYZ_values  = []
        filtered_X = []
        filtered_Z = []
        filtered_Z = []
        
        for joint in SkeletonJoints:
            joint_number = joint.value
            #or 'WRIST' in joint.name
            
            if 'HAND' in joint.name  or 'THUMB' in joint.name or 'FOOT' in joint.name or 'ANKLE' in joint.name:
                _N=2
                _fc=15
                _fs=fs
            else:
                _N=N
                _fc=fc
                _fs=fs
            
            X, Y, Z = sm.filter_signal(noisy_raw_XYZ[0, joint_number, :],
                                       noisy_raw_XYZ[1, joint_number, :],
                                       noisy_raw_XYZ[2, joint_number, :],
                                       N=_N, fc=_fc, fs=_fs)
            
            
            if len(filtered_X) == 0:
                filtered_X = X
                filtered_Y = Y
                filtered_Z = Z
            else:
                filtered_X = np.dstack([filtered_X, X])
                filtered_Y = np.dstack([filtered_Y, Y])
                filtered_Z = np.dstack([filtered_Z, Z])
        
     
        filtered_X = np.transpose(filtered_X)
        filtered_Y = np.transpose(filtered_Y)
        filtered_Z = np.transpose(filtered_Z)
        
        stacked_filtered_XYZ_values = np.stack([filtered_X[:,:,0], filtered_Y[:,:,0], filtered_Z[:,:,0]])  
        
        return stacked_filtered_XYZ_values
    
    
    def calulate_CD_and_NM(self, rebuild=False, save_pngs=False):
        """
        This method calculates the cosine distance and normalised magnitude between joints in a skeleton. 
        It first checks if the cached files for the cosine distance and normalised magnitude exist. 
        If they do not exist or if the rebuild flag is set to True, it calculates the cosine distance and normalised magnitude between joints in the skeleton. 
        It then saves the results in a numpy file and a csv file. If the save_pngs flag is set to True, it also saves the cosine distance and normalised magnitude as png files. 
        Finally, it returns the cosine distance and normalised magnitude.
        
        @param self - the object instance
        @param rebuild - a flag to indicate whether to rebuild the cache files
        @param save_pngs - a flag to indicate whether to save the cosine distance
        """
        scaled = ''
        if self._scale_skeletons:
            scaled = 'scaled_'
                
        joint_set = 'no_hands_'
        ref_set = 'CoM_'
        cached_CD_file_name = os.path.join(self._root_path, 
                                           (self._dataset_prefix +
                                            str(self._part_id) + '_' +
                                            scaled +
                                            joint_set +
                                            ref_set +
                                            'CD_' +                                            
                                            self._movement + '.npy'))
        
        cached_NM_file_name = os.path.join(self._root_path,
                                           (self._dataset_prefix +
                                            str(self._part_id) + '_' +
                                            scaled +
                                            joint_set +
                                            ref_set +
                                            'NM_' +                                            
                                            self._movement + '.npy'))
        
        cd_cache_file_exists = os.path.exists(cached_CD_file_name)
        nm_cache_file_exists = os.path.exists(cached_NM_file_name)
        
        if not cd_cache_file_exists or not nm_cache_file_exists or rebuild:
            cosine_distance = []
            normalised_magnitude = []

            for skel_idx in tqdm(range(np.shape(self.stacked_filtered_XYZ_values)[2])):
                skel = self.stacked_filtered_XYZ_values[:,:,skel_idx]
                
                cd_row = []
                nm_row = []
                col_names = []
                for ref_joint in RefernceTorsoJoints_COM: #RefernceTorsoJoints_HEAD RefernceTorsoJoints_NECK RefernceTorsoJoints_COM
                    for skel_joint in SkeletonJoints_no_hands_or_feet: #SkeletonJoints
                        cd = sm.cosine_distance_between_joints(skel[:, ref_joint.value], 
                                                               skel[:, skel_joint.value])
                        col_names.append(ref_joint.name + '_' + skel_joint.name)
                        # cd_csv_rows.append({'ref_joint': ref_joint.name,
                        #                    'skel_joint': skel_joint.name,
                        #                    'cd': cd})
                        
                        cd_row.append(cd)
                        
                        nm = sm.normalised_magnitude_between_joints(skel[:, skel_joint.value],
                                                                    skel[:, ref_joint.value])
                        
                        # nm_csv_rows.append({'ref_joint': ref_joint.name,
                        #                    'skel_joint': skel_joint.name,
                        #                    'nm': nm})
                        
                        nm_row.append(nm)
                        
                cosine_distance.append(cd_row)
                normalised_magnitude.append(nm_row)    
                
            self.cosine_distance = cosine_distance
            self.normalised_magnitude = normalised_magnitude
            
            #save
            np.save(cached_CD_file_name, self.cosine_distance)
            np.save(cached_NM_file_name, self.normalised_magnitude)
            
            # for col in class SkeletonJoints_no_hands_or_feet:
            #     col_names.append(col.name)
            pd.DataFrame(self.cosine_distance, columns=col_names).to_csv(cached_CD_file_name.replace('.npy', '.csv'))
            pd.DataFrame(self.normalised_magnitude, columns=col_names).to_csv(cached_NM_file_name.replace('.npy', '.csv'))


        else:                        
            self.cosine_distance = np.load(cached_CD_file_name)
            print('loading:', cached_CD_file_name, '\n')
            
            self.normalised_magnitude = np.load(cached_NM_file_name)
            print('loading:', cached_NM_file_name, '\n')
            
            
        if save_pngs:
            str_part_id = str(self._part_id)
            plt.figure()
            plt.plot(kinect_recording.cosine_distance)
            plt.title('Cosign distance ' + scaled + ' ' + str_part_id)
            plt.savefig(os.path.join(self._root_path,('CD_' + scaled + 
                                                      ref_set + joint_set +
                                                      str_part_id)))
            plt.close()
            
            
            plt.figure()
            plt.plot(kinect_recording.normalised_magnitude)
            plt.title('Normalised magnitude ' + scaled + ' ' + str_part_id)
            plt.savefig(os.path.join(self._root_path,('NM_' + scaled +
                                                      ref_set + joint_set +
                                                      str_part_id)))
            plt.close()
            
            
            img = pd.DataFrame(self.cosine_distance).values
            #min_max_scaler = preprocessing.MinMaxScaler()
            img_scaled = ((img - img.min()) * (1/(img.max() 
                         - img.min()) * 255)).astype('uint8')
            img_3ch = np.dstack([img_scaled, img_scaled, img_scaled])
            img_3ch_to_save = pil_image.fromarray(img_3ch, mode='RGB')
            img_3ch_to_save.save(cached_NM_file_name.replace('.npy', '.png'))
    
    
    def normalise_skeleton(self, skel_frame):        
        """
        This function normalizes a skeleton frame by subtracting the reference spine base
        from each joint's X, Y, and Z coordinates. If the `_scale_skeletons` flag is set,
        the function scales the skeleton by the scale factor for each joint. The function
        also updates the reference spine base and skeleton frame if they are empty.

        @param self - the class instance
        @param skel_frame - the skeleton frame to normalize
        @return the normalized skeleton frame
        """
        if len(self._ref_spine_base) == 0:
            self._ref_spine_base =  skel_frame.iloc[SkeletonJoints.SPINEBASE.value][['X', 'Y', 'Z']].tolist()
            self._ref_skel_frame =  skel_frame
                
                
        normalised_skel_frame = pd.DataFrame.copy(skel_frame,deep=False)
        # x = 0
        # y = 1
        # z = 2
                
        for i, joint in enumerate(SkeletonJoints): #HierarchyForBodySegments
            joint_name = joint.name
            joint_number = joint.value
                
            normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('X')] = (skel_frame.iloc[joint_number]['X'] - self._ref_spine_base[X])
            normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('Y')] = (skel_frame.iloc[joint_number]['Y'] - self._ref_spine_base[Y])
            normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('Z')] = (skel_frame.iloc[joint_number]['Z'] - self._ref_spine_base[Z])
        
            if self._scale_skeletons:
                skel_scale = self.get_scale_for_joint(skel_frame, joint_name)
                         
                normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('X')] = (normalised_skel_frame.iloc[joint_number]['X'] * skel_scale)
                normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('Y')] = (normalised_skel_frame.iloc[joint_number]['Y'] * skel_scale)
                normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('Z')] = (normalised_skel_frame.iloc[joint_number]['Z'] * skel_scale)
        
            if joint_number == 24:
                break
       
        self._frame_count +=1
        
        
        return normalised_skel_frame
    
    
    def _load_skel_file(self, skel_file_path):
        """
        Load a skeleton file from a given path and return the skeleton frame and the raw XYZ values.
        @param self - the class instance
        @param skel_file_path - the path to the skeleton file
        @return skel_frame - the skeleton frame
        @return raw_XYZ - the raw XYZ values
        """

        """ debug skelfile """
        #print(skel_file_path)
        
        #replace with array of skeletons
        columns = ['Joint', 'Tracked', 'X', 'Y', 'Z', 'px_X', 'px_Y']
        skel_frame = pd.read_csv(skel_file_path, delimiter=' ', header=None, nrows=25, names=columns, index_col=0)
        

        ''' Normalise '''
        skel_frame = self.normalise_skeleton(skel_frame)
        
        #Add CoM
        tmp_CoM = self.calulate_CoM_position(skel_frame)

        tmp_CoM_row = {'Tracked':'Tracked',
                       'X':tmp_CoM[0],
                       'Y':tmp_CoM[1],
                       'Z':tmp_CoM[2],
                       'px_X':0,
                       'px_Y':0}
        
        df_CoM_row = pd.DataFrame(tmp_CoM_row, index=['CoM'])
        
        skel_frame = skel_frame.append(df_CoM_row)
        
        X = skel_frame['X']
        Y = skel_frame['Y']
        Z = skel_frame['Z']
        raw_XYZ = np.stack([X.values, Y.values, Z.values])
        
        return skel_frame, raw_XYZ
    
    
    def calulate_CoM_position(self, skel_frame):
        """
        Calculate the center of mass (CoM) position of a skeleton frame.
        @param skel_frame - the skeleton frame
        @return The CoM position as a list.
        """
        # _X = 2
        # _Y = 3
        # _Z = 4
    
        spine_base = np.stack([skel_frame['X'][SkeletonJoints.SPINEMID.value], skel_frame['Y'][SkeletonJoints.SPINEMID.value], skel_frame['Z'][SkeletonJoints.SPINEMID.value]])
        hip_left = np.stack([skel_frame['X'][SkeletonJoints.HIPLEFT.value], skel_frame['Y'][SkeletonJoints.HIPLEFT.value], skel_frame['Z'][SkeletonJoints.HIPLEFT.value]])
        hip_right = np.stack([skel_frame['X'][SkeletonJoints.HIPRIGHT.value], skel_frame['Y'][SkeletonJoints.HIPRIGHT.value], skel_frame['Z'][SkeletonJoints.HIPRIGHT.value]])
    
        x_mean = np.mean([spine_base[0],hip_left[0],hip_right[0]])
        y_mean = np.mean([spine_base[1],hip_left[1],hip_right[1]])
        z_mean = np.mean([spine_base[2],hip_left[2],hip_right[2]])
    
        CoM = np.stack([x_mean,y_mean,z_mean])
    
        #com = spine_base
    
        return CoM.tolist()
    
    
    def calulate_skeleton_angles_from_stacked_XYZ_values(self, stacked_filtered_XYZ_values):
        """
        This function calculates the angles of various joints in the human body from the stacked XYZ values of the joints. 
        The function returns an array of angles for each joint. 
        """
        angles_list = []
        #X = 0
        #Y = 1
        #Z = 2
        
        tmp_foot_left_joint = np.stack([np.mean(stacked_filtered_XYZ_values[X][SkeletonJoints.FOOTLEFT.value]),
                                        np.mean(stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTLEFT.value]),
                                        np.mean(stacked_filtered_XYZ_values[Z][SkeletonJoints.FOOTLEFT.value])])
            
        tmp_foot_left_joint = np.stack([np.mean(stacked_filtered_XYZ_values[X][SkeletonJoints.FOOTRIGHT.value]),
                                        np.mean(stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTRIGHT.value]),
                                        np.mean(stacked_filtered_XYZ_values[Z][SkeletonJoints.FOOTRIGHT.value])])
        
        tmp_foot_mean_joint = sm.mean_twin_joint_pos(tmp_foot_left_joint,
                                                     tmp_foot_left_joint)
        
        tmp_ground_plane = np.stack([np.mean(stacked_filtered_XYZ_values[X][SkeletonJoints.COM.value]),
                                           tmp_foot_mean_joint[1],
                                           np.mean(stacked_filtered_XYZ_values[Z][SkeletonJoints.COM.value])])
        
        for i in np.ndindex(stacked_filtered_XYZ_values.shape[2]):        
            tmp_shoulder_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.SHOULDERLEFT.value][i],
                                                stacked_filtered_XYZ_values[Y][SkeletonJoints.SHOULDERLEFT.value][i],
                                                stacked_filtered_XYZ_values[Z][SkeletonJoints.SHOULDERLEFT.value][i]])
            
            tmp_shoulder_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.SHOULDERRIGHT.value][i],
                                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.SHOULDERRIGHT.value][i],
                                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.SHOULDERRIGHT.value][i]])
            
            tmp_spine_shoulder_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.SPINESHOULDER.value][i],
                                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.SPINESHOULDER.value][i],
                                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.SPINESHOULDER.value][i]])
            
            tmp_com_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.COM.value][i],
                                      stacked_filtered_XYZ_values[Y][SkeletonJoints.COM.value][i],
                                      stacked_filtered_XYZ_values[Z][SkeletonJoints.COM.value][i]])
            
           
            tmp_elbow_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ELBOWLEFT.value][i],
                                             stacked_filtered_XYZ_values[Y][SkeletonJoints.ELBOWLEFT.value][i],
                                             stacked_filtered_XYZ_values[Z][SkeletonJoints.ELBOWLEFT.value][i]])
                    
            tmp_elbow_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ELBOWRIGHT.value][i],
                                              stacked_filtered_XYZ_values[Y][SkeletonJoints.ELBOWRIGHT.value][i],
                                              stacked_filtered_XYZ_values[Z][SkeletonJoints.ELBOWRIGHT.value][i]])
            
            tmp_wrist_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.WRISTLEFT.value][i],
                                             stacked_filtered_XYZ_values[Y][SkeletonJoints.WRISTLEFT.value][i],
                                             stacked_filtered_XYZ_values[Z][SkeletonJoints.WRISTLEFT.value][i]])
                    
            tmp_wrist_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.WRISTRIGHT.value][i],
                                              stacked_filtered_XYZ_values[Y][SkeletonJoints.WRISTRIGHT.value][i],
                                              stacked_filtered_XYZ_values[Z][SkeletonJoints.WRISTRIGHT.value][i]])
            
            
            
            tmp_hip_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.HIPLEFT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.HIPLEFT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.HIPLEFT.value][i]])
            
            tmp_hip_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.HIPRIGHT.value][i],
                                  stacked_filtered_XYZ_values[Y][SkeletonJoints.HIPRIGHT.value][i],
                                  stacked_filtered_XYZ_values[Z][SkeletonJoints.HIPRIGHT.value][i]])
            
            tmp_knee_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.KNEELEFT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.KNEELEFT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.KNEELEFT.value][i]])
            
            tmp_knee_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.KNEERIGHT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.KNEERIGHT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.KNEERIGHT.value][i]])
            
            tmp_ankle_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ANKLELEFT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.ANKLELEFT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.ANKLELEFT.value][i]])
            
            tmp_ankle_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ANKLERIGHT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.ANKLERIGHT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.ANKLERIGHT.value][i]])
            
            tmp_heel_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ANKLELEFT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTLEFT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.ANKLELEFT.value][i]])
            
            tmp_heel_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ANKLERIGHT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTRIGHT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.ANKLERIGHT.value][i]])
            
            tmp_heel_mean_joint = sm.mean_twin_joint_pos(tmp_heel_left_joint, tmp_heel_right_joint)
            
            tmp_ankle_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ANKLELEFT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.ANKLELEFT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.ANKLELEFT.value][i]])
            
            tmp_ankle_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ANKLERIGHT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.ANKLERIGHT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.ANKLERIGHT.value][i]])
            
            
            tmp_foot_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.FOOTLEFT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTLEFT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.FOOTLEFT.value][i]])
            
            tmp_foot_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.FOOTRIGHT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTRIGHT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.FOOTRIGHT.value][i]])
            
            body_com_angle = sm.get_angle_between_three_joints(tmp_com_joint, 
                                                               tmp_heel_mean_joint, 
                                                               tmp_ground_plane)                          
                                      
            body_lean_angle = sm.get_angle_between_three_joints(tmp_spine_shoulder_joint, 
                                                               tmp_heel_mean_joint, 
                                                               tmp_ground_plane)
            
            knee_angle_left = sm.get_angle_between_three_joints(tmp_hip_left_joint, 
                                                                tmp_knee_left_joint,
                                                                tmp_ankle_left_joint)
            
            knee_angle_right = sm.get_angle_between_three_joints(tmp_hip_right_joint, 
                                                                 tmp_knee_right_joint,
                                                                 tmp_ankle_right_joint)
            
            hip_angle_left = sm.get_angle_between_three_joints(tmp_shoulder_left_joint, 
                                                               tmp_hip_left_joint,            
                                                               tmp_knee_left_joint)
            
            hip_angle_right = sm.get_angle_between_three_joints(tmp_shoulder_right_joint, 
                                                                tmp_hip_right_joint,            
                                                                tmp_knee_right_joint)
            
            
            elbow_angle_left = sm.get_angle_between_three_joints(tmp_shoulder_left_joint,
                                                                 tmp_elbow_left_joint,            
                                                                 tmp_wrist_left_joint)
            
            elbow_angle_right = sm.get_angle_between_three_joints(tmp_shoulder_right_joint,
                                                                  tmp_elbow_right_joint,            
                                                                  tmp_wrist_right_joint)
                        
            
            
            armpit_angle_left = sm.get_angle_between_three_joints(tmp_hip_left_joint,
                                                                  tmp_shoulder_left_joint,            
                                                                  tmp_elbow_left_joint)
            
            armpit_angle_right = sm.get_angle_between_three_joints(tmp_hip_right_joint,
                                                                   tmp_shoulder_right_joint,            
                                                                   tmp_elbow_right_joint)
            
            ankle_angle_left = sm.get_angle_between_three_joints(tmp_foot_left_joint,
                                                                  tmp_heel_left_joint,            
                                                                  tmp_knee_left_joint)
            
            ankle_angle_right = sm.get_angle_between_three_joints(tmp_foot_right_joint,
                                                                  tmp_heel_right_joint,            
                                                                  tmp_knee_right_joint)
                  
            '''                    
                BODY_COM_ANGLE = 1
                BODY_LEAN_ANGLE = 2
                KNEELEFT_ANGLE = 3
                KNEERIGHT_ANGLE = 4
                HIPLEFT_ANGLE = 5
                HIPRIGHT_ANGLE = 6
                ELBOWLEFT_ANGLE = 7
                ELBOWLEFT_ANGLE = 8
                ARMPITLEFT_ANGLE = 9
                ARMPITLEFT_ANGLE = 10
                ANKLELEFT_ANGLE = 11
                ANKLELRIGHT_ANGLE = 12
            '''
            
            angles_row = np.hstack([self._part_id,
                                    body_com_angle,
                                    body_lean_angle,
                                    knee_angle_left,
                                    knee_angle_right,
                                    hip_angle_left,
                                    hip_angle_right,
                                    elbow_angle_left,
                                    elbow_angle_right,
                                    armpit_angle_left,
                                    armpit_angle_right,
                                    ankle_angle_left,
                                    ankle_angle_right])
            
            if len(self._labels) != 0: 
                arr_labels = np.array(self._labels)[0]
                self.load_sway_metrics()
                angles_row = np.concatenate((angles_row, np.array(self.sway_metrics.iloc[0,10:]), arr_labels))
                
            
            angles_list.append(angles_row)
        
        return np.array(angles_list)
        
    
    
    def calculate_walked_skel_angles(self):
        """
        Calculate the angles between three joints in a skeleton. 
        The function takes no arguments and returns nothing. 
        It calculates the angles between three joints in a skeleton and stores the results in a pandas dataframe. 
        The dataframe has columns for each combination of joints and the angles between them. 
        
        The function uses the get_3d_angle_between_three_joints function from the metrics.py
        """
        columns = []
        for combination in WalkedSkelAnglesIn3s:
        #for combination in WalkedSkelAngles:
            columns = np.hstack([columns, (combination.name + '_SP')])
            columns = np.hstack([columns, (combination.name + '_FP')])
            columns = np.hstack([columns, (combination.name + '_TP')])

        skel_walked_angels = []
        X = 0
        Y = 1
        Z = 2
        for i in range(self.stacked_filtered_XYZ_values.shape[2]):
            skel_angles_row = np.array([])
            for combination in WalkedSkelAnglesIn3s:
            #for combination in WalkedSkelAngles:
                j1  = np.stack([1 * self.stacked_filtered_XYZ_values[X][combination.value[0]][i],
                                1 * self.stacked_filtered_XYZ_values[Y][combination.value[0]][i],
                                1 * self.stacked_filtered_XYZ_values[Z][combination.value[0]][i]])
                
                j2  = np.stack([1 * self.stacked_filtered_XYZ_values[X][combination.value[1]][i],
                                1 * self.stacked_filtered_XYZ_values[Y][combination.value[1]][i],
                                1 * self.stacked_filtered_XYZ_values[Z][combination.value[1]][i]])
                
                j3  = np.stack([1 * self.stacked_filtered_XYZ_values[X][combination.value[2]][i],
                                1 * self.stacked_filtered_XYZ_values[Y][combination.value[2]][i],
                                1 * self.stacked_filtered_XYZ_values[Z][combination.value[2]][i]])
        
                
                euler_angle_3d = sm.get_3d_angle_between_three_joints(j1, j2, j3, deg=True)
                
                if len(skel_angles_row) == 0:
                    skel_angles_row = euler_angle_3d
                else:
                    skel_angles_row = np.hstack([skel_angles_row, euler_angle_3d])
                
            skel_walked_angels.append(skel_angles_row)
            
        pd_skel_walked_angels = pd.DataFrame(skel_walked_angels, columns=columns)
                        
        self.walked_skel_angles = pd_skel_walked_angels