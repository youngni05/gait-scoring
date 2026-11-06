#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:46:23 2019

@author: Sean Maudsley-Barton

A set of utilities that help to calculate common sway metrics
"""

import numpy as np
import pandas as pd
import os

import math
import scipy
from scipy import signal
from scipy.spatial import distance
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from enum import Enum        

#%%
class SOTTrial(Enum):
    """
    This is an enumeration class that defines the different types of SOT (Sensory Organization Test) trials. 
    It has six different types of trials, each with a unique integer value.
    """
    Eyes_Open_Fixed_Surround_and_Support = 1
    Eyes_Closed_Fixed_Support = 2
    Open_Sway_Referenced_Surrond = 3
    Eyes_Open_Sway_Referenced_Support = 4
    Eyes_Closed_Sway_Referenced_Support = 5
    Eyes_Open_Sway_Referenced_Surround_and_Support = 6


class SwayMetric(Enum):
    """
    This is an enumeration class for sway metric. It defines a set of named constants
    that represent the different types of sway metrics. Each constant has an integer value
    associated with it.
    """
    ALL = 0
    RDIST = 1
    RDIST_ML = 2
    RDIST_AP = 3
    MDIST = 4
    MDIST_ML = 5
    MDIST_AP = 6
    TOTEX = 7
    TOTEX_ML = 8
    TOTEX_AP = 9
    MVELO = 10
    MVELO_ML = 11
    MVELO_AP = 12
    MFREQ = 13
    MFREQ_ML = 14
    MFREQ_AP = 15
    AREA_CE = 16
    FRAC_DIM = 17
    ROMBERG_RATIO = 18
    ROMBERG_RATIO_FOAM = 19
    
    
class DeviceType(Enum):
    """
    This is a simple enumeration class that defines two device types: BALANCE_MASTER and KINECT. 
    These device types are used in the sway metric.
    """
    BALANCE_MASTER = 1
    KINECT = 2

class SwayGroup(Enum):
    """
    This is an enumeration class for sway groups. It defines different sway groups as enum values. 
    Each enum value is assigned a unique integer value. 
    The sway groups are:
    """
    All = 0
    All_healthy = 1
    All_fallers = 2
    
    Young = 3
    Middle = 4
    Old = 5
    
    Faller_by_history = 6
    Faller_by_history_single = 7
    Faller_by_history_muliple = 8
    Faller_by_miniBEStest = 9
    
    Young_vs_old = 10
    Old_vs_all_fallers = 11
    Old_vs_single_fallers = 12
    Young_vs_all_fallers = 13
    Young_and_Middle = 14
    

"""
This code defines a set of constants that map to the indices of the joints in a skeleton. 
Each constant represents a specific joint in the skeleton, as follows:
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

"""
This associates X,Y,X with 0, 1, and 2.
This makes working with 3D arrays that represent the movement easier.
"""
X = 0
Y = 1
Z = 2

#%%
def normalise_skeleton(skel_frame, spine_base_joint):
    """
    Normalize the skeleton frame by subtracting the spine base joint from each joint
    and scaling the result by 100.
    @param skel_frame - the skeleton frame to normalize
    @param spine_base_joint - the spine base joint
    @return The normalized skeleton frame
    """
    
    normalised_skel_frame = np.copy(skel_frame)
    # x = 0
    # y = 1
    # z = 2

    for i in range(skel_frame.shape[0]):
        normalised_skel_frame[i][2] =  str(100*(float(skel_frame[i][2]) - spine_base_joint[X]))
        normalised_skel_frame[i][3] =  str(100*(float(skel_frame[i][3]) - spine_base_joint[Y]))
        normalised_skel_frame[i][4] =  str(100*(float(skel_frame[i][4]) - spine_base_joint[Z]))

    return normalised_skel_frame


def calculate_com(skelFrame):
    """
    Calculate the center of mass (COM) of a skeleton frame.
    @param skelFrame - the skeleton frame
    @return The center of mass of the skeleton frame as a list.
    """
    _X = 2
    _Y = 3
    _Z = 4

    spine_shoulder = np.stack([float(skelFrame[SPINESHOULDER, _X]), float(skelFrame[SPINESHOULDER, _Y]), float(skelFrame[SPINESHOULDER, _Z])], axis=0)
    spine_base = np.stack([float(skelFrame[SPINEBASE, _X]), float(skelFrame[SPINEBASE, _Y]), float(skelFrame[SPINEBASE, _Z])], axis=0)
    spine_mid = np.stack([float(skelFrame[SPINEMID, _X]), float(skelFrame[SPINEMID, _Y]), float(skelFrame[SPINEMID, _Z])], axis=0)
    hip_left = np.stack([float(skelFrame[HIPLEFT, _X,]), float(skelFrame[HIPLEFT, _Y,]), float(skelFrame[HIPLEFT, _Z])], axis=0)
    hip_right = np.stack([float(skelFrame[HIPRIGHT, _X]), float(skelFrame[HIPRIGHT, _Y]), float(skelFrame[HIPRIGHT, _Z])], axis=0)
    
    xMean = np.mean([spine_mid[0],hip_left[0],hip_right[0]])
    yMean = np.mean([spine_mid[1],hip_left[1],hip_right[1]])
    zMean = np.mean([spine_mid[2],hip_left[2],hip_right[2]])
    
    com = np.stack([xMean,yMean,zMean],axis=0)

    return com.tolist()


def euclidean_distance_between_joints(j1, j2):
    """
    Calculate the Euclidean distance between two joints.
    @param j1 - the first joint
    @param j2 - the second joint
    @return The Euclidean distance between the two joints.
    """
    ed = np.sqrt(np.square(j1[X] - j2[X]) + np.square(j1[Y] - j2[Y]) + np.square(j1[Z] - j2[Z]))

    return ed


def cosine_distance_between_joints(j1, j2):
    """
    Calculate the cosine distance between two joint vectors.
    @param j1 - the first joint vector
    @param j2 - the second joint vector
    @return The cosine distance between the two joint vectors.
    """
    cd = distance.cosine(j1, j2)
    
    return cd


def normalised_magnitude_between_joints(j, j_ref):
    """
    Calculate the normalised magnitude between two joints.
    @param j - the first joint
    @param j_ref - the second joint
    @return The normalised magnitude between the two joints.
    """
    mag_j = np.linalg.norm(j)
    mag_j_ref = np.linalg.norm(j_ref)
    
    nn = mag_j / mag_j_ref
    
    return nn


def get_joint_XYZ(skel_frame_row):
    """
    Given a row of a skeleton frame, extract the X, Y, and Z coordinates of the joint.
    @param skel_frame_row - a row of a skeleton frame
    @return a list containing the X, Y, and Z coordinates of the joint
    """
    x = float(skel_frame_row[2])
    y = float(skel_frame_row[3])
    z = float(skel_frame_row[4])

    return [x, y, z]


def eulidean_distance_2d(j1, j2):
    """
    Calculate the Euclidean distance between two 2D points.
    @param j1 - the first point
    @param j2 - the second point
    @return The Euclidean distance between the two points.
    """
    ed = np.sqrt(np.square(j1[0] - j2[0]) + np.square(j1[1] - j2[1]))
    return ed

def get_angle(j1, j2, j3):
    """
    Calculate the angle between three joints.
    @param j1 - the first joint
    @param j2 - the second joint
    @param j3 - the third joint
    @return The angle between the three joints in degrees.
    """
    ba = j1 - j2
    bc = j3 - j2

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    rad_angle = np.arccos(cosine_angle)
    deg_angle = np.degrees(rad_angle) 

    return deg_angle


def get_AP_angle_between_three_joints_matrix(j1, j2, j3):
    """
    Calculate the angle between three joints using a matrix.
    @param j1 - the first joint
    @param j2 - the second joint
    @param j3 - the third joint
    @return The angle between the three joints in degrees.
    """
    
    l1 =  np.sqrt(np.square(j1[2] - j2[2]))
    l2 =  np.sqrt(np.square(j2[2] - j3[2]))
    
    rad_between_three_joints = np.arccos(l2/l1)
    deg_between_three_joints = np.degrees(rad_between_three_joints) 
    
    return deg_between_three_joints

def get_2d_angle_between_three_joints(j1, j2, j3, plane='SP', deg=True):
    """
    Calculate the angle between three joints in 2D space.
    @param j1 - the first joint
    @param j2 - the second joint
    @param j3 - the third joint
    @param plane - the plane in which the angle is being calculated
    @param deg - whether the angle should be returned in degrees or radians
    @return The angle between the three joints in the specified plane.
    """
    
    if plane == 'SP':
        co_plane = [Z, Y] # SI, AP | Sagital Plane
    elif plane == 'FP':
        co_plane = [X, Y] # [X, Y] # ML, SI | Frontal Plane
    elif plane == 'TP':
        co_plane = [X, Z] # [X, Z] # ML, AP | Transverse Plane
    
    a = eulidean_distance_2d([j1[co_plane[0]], j1[co_plane[1]]], [j2[co_plane[0]], j2[co_plane[1]]])
    b = eulidean_distance_2d([j2[co_plane[0]], j2[co_plane[1]]], [j3[co_plane[0]], j3[co_plane[1]]]) #j2, j3)
    c = eulidean_distance_2d([j3[co_plane[0]], j3[co_plane[1]]], [j1[co_plane[0]], j1[co_plane[1]]]) #j3, j1)
    
    angle_C = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
    
    if deg:
        angle_C = np.degrees(angle_C)
    
    return angle_C


def get_3d_angle_between_three_joints(j1, j2, j3, deg=True):
    """
    Calculate the 3D angle between three joints. This function calls another function
    to calculate the 2D angle between three joints in three different planes (Sagittal (SP), Frontal(FP), Transverse(TP)).
    @param j1 - the first joint
    @param j2 - the second joint
    @param j3 - the third joint
    @param deg - whether to return the angle in degrees or radians (default is degrees)
    @return The 3D angle between the three joints
    """
    SP_angle = get_2d_angle_between_three_joints(j1, j2, j3, plane='SP', deg=deg)
    FP_angle = get_2d_angle_between_three_joints(j1, j2, j3, plane='FP', deg=deg)
    TP_angle = get_2d_angle_between_three_joints(j1, j2, j3, plane='TP', deg=deg)
    
    #eular_angle_3d = np.hstack([x_YZ_euler, y_XZ_euler, z_XY_euler])
    SP_FP_TP_angle_3d = np.hstack([SP_angle, FP_angle, TP_angle])
    return SP_FP_TP_angle_3d

#%%
def get_unique_values(full_list):
    """
    This function takes a list and returns a new list containing only the unique values from the original list, 
    while preserving the order of the original list.

    @param full_list - the list to be processed
    @return a new list containing only the unique values from the original list, while preserving the order of the original list.
    """
    unnique_list = []
    
    for item in full_list:
        if item not in unnique_list:
            unnique_list.append(item)
            
    return unnique_list


def eigsorted(cov):
    """
    Calculate the eigenvalues and eigenvectors of the covariance matrix.

    @param cov - the covariance matrix
    @return The eigenvalues and eigenvectors of the covariance matrix.
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]

def confidence_ellipse(x, y, ax, n_std=1.96, facecolor='none', return_ellipse=True, **kwargs):
    """
    This function creates a plot of the covariance confidence ellipse of x and y.
    
    @param x - array_like, shape (n, ) - Input data.
    @param y - array_like, shape (n, ) - Input data.
    @param ax - matplotlib.axes.Axes - The axes object to draw the ellipse into.
    @param n_std - float - The number of standard deviations to determine the ellipse's radiuses. 1.96 SD = 95% confidence ellipse
    @param facecolor - string - The color of the ellipse
    @param return_ellipse - boolean - Whether to return the ellipse object or just the ellipse parameters
    @return If return_ellipse is True, returns the ellipse object, height, width, area, and angle. Otherwise, returns height
    """
    if x.size != y.size:
         raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    
    #eigvals, eigvecs = eigsorted(cov)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    eigvals = [
        [eigvals[0], 0],
        [0, eigvals[1]]
        ]
    axes = 2.4478 * np.sqrt(scipy.linalg.svdvals(eigvals))
    height = axes[0] * 2
    width = axes[1] * 2
    area = np.pi * np.prod(axes)                           

    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)), width=width, height=height,
                      angle=angle, facecolor=facecolor, **kwargs)
    
    if return_ellipse == True:    
        return ax.add_patch(ellipse), height, width, area, angle
    else:
        return height, width, area, angle
    

def filter_signal(ML_path, AP_path=[], CC_path=np.array([]), N=2, fc=10, fs=30):
    """
    This function filters a signal using a Butterworth filter. 
    
    @param ML_path - The path of the signal to be filtered in the Mediolateral direction
    @param AP_path - The path of the signal to be filtered in the anterior-posterior direction
    @param CC_path - The path of the signal to be filtered in the cranial-caudal direction
    @param N - The order of the filter - usually 1, 2 or 4
    @param fc - The cut-off frequency of the filter - usually 10 or 6 
    @param fs - The sampling frequency of the signal - for kinect data this is 30
    Wn (Normalize the frequency) = fc / (fs / 2) 
    """
    
    Wn = np.pi * fc / (2 * fs) # Normalize the frequency
    
    #b, a = signal.butter(N, Wn, btype='low', fs=fs)
    b, a = signal.butter(N, Wn, btype='low')
    filtered_ML_path = signal.filtfilt(b, a, ML_path)    
    
    if len(AP_path) != 0:
        filtered_AP_path = signal.filtfilt(b, a, AP_path)
    
    if len(CC_path) != 0:
        filtered_CC_path = signal.filtfilt(b, a, CC_path)

    if len(CC_path) != 0:
        return filtered_ML_path, filtered_AP_path, filtered_CC_path
    if len(AP_path) != 0:
        return filtered_ML_path, filtered_AP_path
    else:
        return filtered_ML_path


def calculate_RD(selected_recording,
               deviceType = DeviceType.KINECT,
               rd_path = '',
               part_id = '',
               SOT_trial_type = ''):
    """
    Calulate resultant distance, 
    that is the distance from each point on the raw CoM path to the 
    mean of the time series ad displays RD path and sway area 
        
    Also, saves AREA_CE image if rd_parh is filled in.

    @param selected_recording - the recording to calculate the RD for
    @param deviceType - the type of device used to record the data
    @param rd_path - the path to save the RD plot
    @param part_id - the participant ID
    @param SOT_trial_type - the type of SOT trial

    @return ML, AP, RD, AREA_CE - the medial-lateral path, anterior-posterior path, resultant distance, and sway area
    """
    
    if deviceType == DeviceType.KINECT:
        ML_path = selected_recording['CoGx'].values.astype(float)
        AP_path = selected_recording['CoGz'].values.astype(float)
        ML_path, AP_path = filter_signal(ML_path, AP_path, fc=8)
        
    elif deviceType == DeviceType.BALANCE_MASTER:
        ML_path = selected_recording['CoGx'].values.astype(float)
        AP_path = selected_recording['CoGy'].values.astype(float)

    mean_ML = np.mean(ML_path)
    mean_AP = np.mean(AP_path)

    ML =  np.abs(np.subtract(ML_path, mean_ML))
    AP =  np.abs(np.subtract(AP_path, mean_AP))
    #ML = np.sqrt(np.square(np.subtract(ML_path, mean_ML)))
    #AP = np.sqrt(np.square(np.subtract(AP_path, mean_AP)))
    
    #get hypotinuse of ML_RD and AP_RD
    RD = np.sqrt(np.add(np.square(ML),np.square(AP)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-8, 8])
    ax.set_ylim([-8, 8])
    
    elp, width, height, AREA_CE = confidence_ellipse(ML_path, AP_path, ax, n_std=1.96, edgecolor='red')
    ax.scatter(ML_path, AP_path, s=3)
    
    AREA_CE = round(AREA_CE, 2)
    
    ax.set_title(str.replace(SOT_trial_type, '-', ' ') + ' ' + part_id + ' CoM 95% CE' + 
                  '\nW:' + str(round(width, 2)) + ' cm' + 
                  ' x ' + 
                  'H:' + str(round(height, 2)) + ' cm' + 
                  ' Ar:' + str(AREA_CE) + ' cm sq')
    
    ax.set_aspect('equal')

    if rd_path != '':
        plt.savefig(rd_path)
    
    plt.show()
        
    fig = plt.figure()
    ax = fig.add_subplot(111)

    elp, width, height, AREA_CE = confidence_ellipse(ML_path, AP_path, ax, n_std=1.96, edgecolor='red')
    ax.scatter(ML_path, AP_path, s=3)
    
    AREA_CE = round(AREA_CE, 2)
    ax.set_title(str.replace(SOT_trial_type, '-', ' ') + ' ' + part_id + ' CoM 95% CE' + 
                 '\nW:' + str(round(width, 2)) + ' cm' + 
                 ' x ' + 
                 'H:' + str(round(height, 2)) + ' cm' + 
                 ' Ar:' + str(AREA_CE) + ' cm sq')
    
    ax.set_aspect('equal')
    
    if rd_path != '':
        detailed_rd_path = rd_path.replace('confidence_ellipse', 'confidence_ellipse_detailed')
        plt.savefig(detailed_rd_path)

    plt.show()

    return ML, AP, RD, AREA_CE


def calculate_RD_3D(selected_recording,
                    deviceType = DeviceType.KINECT,):
    """
    Calculate the resultant distance of a selected recording based on the device type.

    @param selected_recording - the recording to calculate the resultant distance for
    @param deviceType - the type of device used to record the data
    @return a tuple containing the medial-lateral resultant distance, anterior-posterior resultant distance, 
    up-down resultant distance, and the overall resultant distance
    """
    
    if deviceType == DeviceType.KINECT:
        ML_path = selected_recording['CoGx'].values.astype(float)
        AP_path = selected_recording['CoGz'].values.astype(float)
        UD_path = selected_recording['CoGy'].values.astype(float)
        ML_path, AP_path = filter_signal(ML_path, AP_path)
    elif deviceType == DeviceType.BALANCE_MASTER:
        ML_path = selected_recording['CoGx'].values.astype(float)
        AP_path = selected_recording['CoGy'].values.astype(float)

    mean_ML = np.mean(ML_path)
    mean_AP = np.mean(AP_path)
    mean_UD = np.mean(UD_path)

    ML_RD = np.sqrt(np.square(np.subtract(ML_path, mean_ML)))
    AP_RD = np.sqrt(np.square(np.subtract(AP_path, mean_AP)))
    UD_RD = np.sqrt(np.square(np.subtract(UD_path, mean_UD)))
    RD = np.sqrt(np.square(np.subtract(ML_path, mean_ML)) + np.square(np.subtract(AP_path, mean_AP)) + np.square(np.subtract(UD_path, mean_UD)))

    return ML_RD, AP_RD, UD_RD, RD


def calculate_TOTEX(ML, AP):
    """
    Calculate the TOTEX (Total Excursion) and FD (Fractal Dimension) given the ML (Medio-Lateral) and AP (Anterior-Posterior) values.

    @param ML - the Medio-Lateral values
    @param AP - the Anterior-Posterior values
    @return a tuple containing the ML_TOTEX, AP_TOTEX, TOTEX, and FD
    """
    arr_ML_diff = []
    arr_AP_diff = []
    arr_tot_diff = []

    ML_TOTEX = 0
    AP_TOTEX = 0
    TOTEX = 0

    for step in range(1, len(AP)):
        ML_curr = ML[step]
        AP_curr = AP[step]

        ML_prev = ML[step - 1]
        AP_prev = AP[step - 1]
        
        ML_diff = abs(ML_curr - ML_prev)
        AP_diff = abs(AP_curr - AP_prev)
        #ML_diff = np.sqrt(np.square(ML_curr - ML_prev))
        #AP_diff = np.sqrt(np.square(AP_curr - AP_prev))

        arr_ML_diff.append(ML_diff)
        arr_AP_diff.append(AP_diff)
        
        # get hypotinuse of ML and AP diff
        RD_diff = np.sqrt(np.add(np.square(ML_diff), np.square(AP_diff)))
        arr_tot_diff.append(RD_diff)

    ML_TOTEX = np.sum(arr_ML_diff)
    AP_TOTEX = np.sum(arr_AP_diff)
    TOTEX = np.sum(arr_tot_diff)
    
    N = len(AP)
    d = max(arr_tot_diff)
    FD = np.log(N) / np.log((N * d) / TOTEX)

    return ML_TOTEX, AP_TOTEX, TOTEX, FD
    

def calculate_sway_from_recording(selected_recording,
                                  selected_recording_name,
                                  pID,
                                  age,
                                  sex,
                                  SOT_trial_type,
                                  tNum,
                                  swayMetric = SwayMetric.ALL,
                                  deviceType = DeviceType.KINECT,
                                  impairment_self = 'healthy',
                                  impairment_confedence = 'healthy',
                                  impairment_clinical = 'healthy',
                                  impairment_stats = 'healthy',
                                  dp = -1,
                                  rd_path = '',
                                  start = 0,
                                  end = 600):
    

    """
    Calculates sway from Kinect or Balance master recordings. 
    It takes in a selected recording, recording name, patient ID, age, sex, SOT trial type, trial number, sway metric, device type (Kinect or Balance Master), 
    + impairment values. 
    It then calculates sway metrics based on the recording and returns the results.

    @param selected_recording - the recording to calculate sway from
    @param selected_recording_name - the name of the recording
    @param pID - the patient ID
    @param age - the age of the patient
    @param sex - the sex of the patient
    @param SOT_trial_type - the type of SOT trial
    @param tNum - the trial number
    @param swayMetric - the metric to use for calculating sway (default is SwayMetric.ALL)
    """
    
    cliped_recording = selected_recording[start : end]

    ML, AP, RD, AREA_CE = calculate_RD(cliped_recording, deviceType, rd_path, pID, SOT_trial_type)
    recording_length = len(RD)

    #--mean DIST and rms DIST
    MDIST_ML = np.sum(ML) / recording_length
    MDIST_AP = np.sum(AP) / recording_length
    MDIST = np.sum(RD) / recording_length

    RDIST_ML = np.sqrt(np.sum(np.square(ML) / recording_length))
    RDIST_AP = np.sqrt(np.sum(np.square(AP) / recording_length))
    RDIST = np.sqrt(np.sum(np.square(RD) / recording_length))

    #--Total Excursion - TOTEX
    TOTEX_ML, TOTEX_AP, TOTEX, FD = calculate_TOTEX(ML, AP)

    #--Mean Velocity - MVELO
    if deviceType == DeviceType.KINECT:
        T = recording_length / 30
    elif  deviceType == DeviceType.BALANCE_MASTER:
        T = recording_length / 100

    MVELO_ML = TOTEX_ML / T
    MVELO_AP = TOTEX_AP / T
    MVELO = TOTEX / T

    #--Mean Fequency - MFREQ
    MFREQ_ML = MVELO_ML / (4*(np.sqrt(2 * MDIST_ML)))
    MFREQ_AP = MVELO_AP / (4*(np.sqrt(2 * MDIST_AP)))
    MFREQ = MVELO / (2 * np.pi * MDIST)


    #TOTEX MVELO RDIST
    if swayMetric == SwayMetric.RDIST:
        swayVal = RDIST
    elif swayMetric == SwayMetric.RDIST_ML:
        swayVal = RDIST_ML
    elif swayMetric == SwayMetric.RDIST_AP:
        swayVal = RDIST_AP

    elif swayMetric == SwayMetric.MDIST:
        swayVal = MDIST
    elif swayMetric == SwayMetric.MDIST_ML:
        swayVal = MDIST_ML
    elif swayMetric == SwayMetric.MDIST_AP:
        swayVal = MDIST_AP

    elif swayMetric == SwayMetric.TOTEX:
        swayVal = TOTEX
    elif swayMetric == SwayMetric.TOTEX_ML:
        swayVal = TOTEX_ML
    elif swayMetric == SwayMetric.TOTEX_AP:
        swayVal = TOTEX_AP

    elif swayMetric == SwayMetric.MVELO:
        swayVal = MVELO
    elif swayMetric == SwayMetric.MVELO_ML:
        swayVal = MVELO_ML
    elif swayMetric == SwayMetric.MVELO_AP:
        swayVal = MVELO_AP

    elif swayMetric == SwayMetric.MFREQ:
        swayVal = MFREQ
    elif swayMetric == SwayMetric.MFREQ_ML:
        swayVal = MFREQ_ML
    elif swayMetric == SwayMetric.MVELO_AP:
        swayVal = MFREQ_AP

    elif swayMetric == SwayMetric.AREA_CE:
        swayVal = AREA_CE
        
    elif swayMetric == SwayMetric.FRAC_DIM:
        swayVal = FD

    tmpSway = []

    if dp != -1:
        swayVal = round(swayVal, dp)
    
    if swayMetric == SwayMetric.ALL:
        tmpSway.append([pID, selected_recording_name, tNum, age, sex, 
                        impairment_self, impairment_confedence, 
                        impairment_clinical, impairment_stats,
                        swayMetric.name,
                        RDIST_ML, RDIST_AP, RDIST,
                        MDIST_ML, MDIST_AP, MDIST,
                        TOTEX_ML, TOTEX_AP, TOTEX,
                        MVELO_ML, MVELO_AP, MVELO,
                        MFREQ_ML, MFREQ_AP, MFREQ,
                        AREA_CE])
    else:
        tmpSway.append([pID, selected_recording_name, tNum, age, sex, 
                        impairment_self, impairment_confedence, 
                        impairment_clinical, impairment_stats,
                        swayMetric.name,
                        swayVal])


    return tmpSway

#%% Balance master Utils
def load_balance_master_file(rootDir,
                             participantID,
                             age,
                             kinectTrialType,
                             trialNumber = 1,
                             swayMetric = SwayMetric.RDIST):
    

    """
    Given a root directory, participant ID, age, kinect trial type, trial number, and sway metric, load the balance master file for the specified trial. 
    
    @param rootDir - the root directory
    @param participantID - the participant ID
    @param age - the age of the participant
    @param kinectTrialType - the type of kinect trial
    @param trialNumber - the trial number
    @param swayMetric - the sway metric
    @return a dataframe of the balance master file and the selected trial
    """

    root = ''
    dirs = []
    columns = ''
    arrayOfRows = []
    dfFinal = pd.DataFrame(arrayOfRows)

    for root, dirs, _ in os.walk(rootDir):
        break

    dirs.sort()

    selected_trial = ''
    for dirName in dirs:
        if participantID in dirName:
            print(dirName)
            rootFilePath = os.path.join(root, dirName, 'cm')

            trialRoot = ''
            trialfiles = []
            for trialRoot, _, trialfiles in os.walk(rootFilePath):
                break

            found = False
            for trial in trialfiles:
                #if 'SOT' in trial and str('T'+ str(kinectTrialType.value)) in trial:
                if 'SOT' in trial and str('C'+ str(kinectTrialType.value)) in trial and str('T'+ str(trialNumber)) in trial:
                    print('Collating:', trialRoot + '/' + trial)
                    selected_trial = trial
                    trialFilePath = os.path.join(trialRoot, trial)

                    bMTrial = pd.read_csv(trialFilePath
                                          , sep="\t")


                    i = 0

                    for row in bMTrial[25:].iterrows():
                        parsedRow = row[1][0]
                        arrRow = parsedRow.split('\t')
                        strArrRow = str.split(arrRow[0])

                        if i == 0:
                            #columns = strArrRow
                            columns = ['DP', 'LF', 'RR', 'SH', 'LR', 'RF', 'CoFx', 'CoFy', 'CoGx', 'CoGy']
                        else:
                            arrayOfRows.append(strArrRow)

                        i += 1

                    found = True
                    break

            if found:
                dfFinal = pd.DataFrame(arrayOfRows, columns=columns)
            else:
                print('can not find ', trialRoot + '/' + trial)

            #break


    return dfFinal, selected_trial

