import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
class DatasetLoader():
    def __init__(self, shape):
        self.shape = shape

    def s_fact(self, SR_FACTOR=2, NB_PAIRS=100, NB_SCALING_STEPS=1): # NB_PAIRS=1500, BATCH_SIZE = 1, NB_SCALING_STEPS = 1
        shape = self.shape
        BLUR_LOW_BIAS = 0.0 
        scale_factors = np.empty(0)
        
        lenpad = np.int(NB_PAIRS / NB_SCALING_STEPS)

        BLUR_LOW = 0.4 # Scaling factors - blurring parameters
        BLUR_HIGH = 0.95 
        if shape[0] * shape[1] <= 50 * 50:
          BLUR_LOW_BIAS = 0.3
        for i in range(NB_SCALING_STEPS):
          temp = np.random.uniform(BLUR_LOW + BLUR_LOW_BIAS, BLUR_HIGH, lenpad)  # Low = 0.4, High = 0.95
          temp = np.sort(temp)
          scale_factors = np.append(scale_factors, temp, axis=0)
          scale_factors = np.around(scale_factors, decimals=5)
        
        scale_factors_pad = np.repeat(scale_factors[-1], abs(NB_PAIRS - len(scale_factors)))
        scale_factors = np.concatenate((scale_factors, scale_factors_pad), axis=0)
        intermidiate_SR_Factors = np.delete(np.linspace(1, SR_FACTOR, NB_SCALING_STEPS + 1), 0)
        intermidiate_SR_Factors = np.around(intermidiate_SR_Factors, decimals=3)

        intermidiate_SR_Factors = np.repeat(intermidiate_SR_Factors, lenpad) # repeat intermidiate_SR_Factors for lenpad
        pad = np.repeat(intermidiate_SR_Factors[-1], abs(len(intermidiate_SR_Factors) - max(len(scale_factors), NB_PAIRS)))  #there could be mismatch as explained before so for padding this step is there
        intermidiate_SR_Factors = np.concatenate((intermidiate_SR_Factors, pad), axis=0)
        
        self.scale_factors=scale_factors
        self.intermidiate_SR_Factors = intermidiate_SR_Factors;
        return self.scale_factors, self.intermidiate_SR_Factors

    