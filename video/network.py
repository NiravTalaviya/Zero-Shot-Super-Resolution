import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, Concatenate, BatchNormalization, MaxPool2D

class ZSSR(Model):
  def __init__(self, NB_CHANNELS=3, FILTERS=64):
    super(ZSSR, self).__init__()
    ALPHA = 0.2
    self.convA = Conv2D(filters=NB_CHANNELS, kernel_size=(3, 3), strides = 1, padding='same')
    self.reluA = LeakyReLU(alpha=ALPHA)  
    # inner layer 1
    self.conv2d_i_1 = Conv2D(filters=FILTERS, kernel_size=(3,3), strides=1, padding='same')
    self.relu_i_1 = LeakyReLU(alpha=ALPHA)
    # inner layer 2
    self.conv2d_i_2 = Conv2D(filters=FILTERS, kernel_size=(3,3), strides=1, padding='same')
    self.relu_i_2 = LeakyReLU(alpha=ALPHA)
    # inner layer 3
    self.conv2d_i_3 = Conv2D(filters=FILTERS, kernel_size=(3,3), strides=1, padding='same')
    self.relu_i_3 = LeakyReLU(alpha=ALPHA)
    # inner layer 4
    self.conv2d_i_4 = Conv2D(filters=FILTERS, kernel_size=(3,3), strides=1, padding='same')
    self.relu_i_4 = LeakyReLU(alpha=ALPHA)
    # inner layer 5
    self.conv2d_i_5 = Conv2D(filters=FILTERS, kernel_size=(3,3), strides=1, padding='same')
    self.relu_i_5 = LeakyReLU(alpha=ALPHA)
    # inner layer 6
    self.conv2d_i_6 = Conv2D(filters=FILTERS, kernel_size=(3,3), strides=1, padding='same')
    self.relu_i_6 = LeakyReLU(alpha=ALPHA)

    # inner layer 7
    self.conv2d_i_7 = Conv2D(filters=FILTERS, kernel_size=(3,3), strides=1, padding='same')
    self.relu_i_7 = LeakyReLU(alpha=ALPHA)

    # inner layer 7
    self.conv2d_i_8 = Conv2D(filters=FILTERS, kernel_size=(3,3), strides=1, padding='same')
    self.relu_i_8 = LeakyReLU(alpha=ALPHA)

    self.convB = Conv2D(filters=NB_CHANNELS, kernel_size=(3,3), strides=1, padding='same', activation='linear')
  
  def call(self, inputs):
    x = self.convA(inputs)
    x = self.reluA(x)

    x = self.conv2d_i_1(x)
    x = self.relu_i_1(x)
    
    x = self.conv2d_i_2(x)
    x = self.relu_i_2(x)

    x = self.conv2d_i_3(x)
    x = self.relu_i_3(x)
    
    x = self.conv2d_i_4(x)
    x = self.relu_i_4(x)

    x = self.conv2d_i_5(x)
    x = self.relu_i_5(x)
    
    x = self.conv2d_i_6(x)
    x = self.relu_i_6(x)

    x = self.conv2d_i_7(x)
    x = self.relu_i_7(x)
    
    x = self.conv2d_i_8(x)
    x = self.relu_i_8(x)

    x = self.convB(x)
    x = tf.keras.layers.add([x, inputs])
    return x;