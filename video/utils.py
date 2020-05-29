import numpy as np
import cv2

import tensorflow as tf

def preprocess(image, scale_fact, scale_fact_inter, CROP_SIZE=[96], 
    NOISE_FLAG=True, noise_std=0.0125):
        image = image.astype(np.float32)
        
        # scale down is sthe inverse of the intermediate scaling factor
        scale_down = 1 / scale_fact_inter 
        # Create hr father by downscaling from the original image
        hr = cv2.resize(image, None, fx=scale_fact, fy=scale_fact, interpolation=cv2.INTER_CUBIC)
        hr = tf.image.random_crop(hr, [min(hr.shape[0], CROP_SIZE[0]), min(hr.shape[1], CROP_SIZE[0]), 3])    
        k = np.random.choice(8)
        hr = np.rot90(hr, k, axes=(0, 1))
        if (k > 3):
            hr = np.fliplr(hr)
            
        hr = tf.keras.preprocessing.image.random_shear(hr, 0.01)
        # Blur lr son
        lr = cv2.resize(hr, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_CUBIC)

        if NOISE_FLAG: 
            lr = add_noise(lr, noise_std)

        # Upsample lr to the same size as hr
        lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)

        X = lr = np.expand_dims(lr, axis=0)  # Expanding dimension
        y = hr = np.expand_dims(hr, axis=0)  # Expanding dimension
        return X, y

def add_noise(image, STD):
    noise = np.random.randn(*image.shape) * STD * 255
    noise = noise.astype('float32') 
    noisy = np.clip((image + noise), 0, 255)
    return noisy

def time_diff(end_time, start_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
def image_generator(self):
      i = 0
      scale_fact, scale_fact_inter = self.s_fact(NB_PAIRS, NB_STEPS)
      while True:
        X, y = self.preprocess(scale_fact[i] + np.round(np.random.normal(0.0, 0.03), decimals=3), scale_fact_inter[i])
        i = i + 1
        yield X, y