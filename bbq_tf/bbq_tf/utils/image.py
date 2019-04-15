import numpy as np



def rgb2gray(rgb):
   """
   Convert an mpimg into grayscale
   :param rgb:
   :return:
   USAGE
   >> import matplotlib.pyplot as plt
   >> import matplotlib.image as mpimg
   >> img = mpimg.imread('image.png')
   >> gray = rgb2gray(img)
   >> plt.imshow(gray, cmap=plt.get_cmap('gray'))
   >> plt.show()
   """
   return np.dot(rgb[:, :, :3], [0.299, 0.587, 0.114])
