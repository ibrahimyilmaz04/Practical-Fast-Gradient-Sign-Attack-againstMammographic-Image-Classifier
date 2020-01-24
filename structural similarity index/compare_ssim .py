from skimage import data, img_as_float
from skimage.measure import compare_ssim
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

ssim = np.zeros((70, 11))
stacknum = 0
for dir in os.listdir('train_db'):
	for file in os.listdir(os.path.join('train_db', dir)):
		img1 = cv2.imread(os.path.join('train_db', dir, file), cv2.IMREAD_GRAYSCALE)
		ssim[stacknum, 0] = compare_ssim(img1, img1)
		for j in range(10):
			img2 = cv2.imread(os.path.join('attack_%02d'%(j+1), dir, file), cv2.IMREAD_GRAYSCALE)
			ssim[stacknum, j + 1] = compare_ssim(img1, img2)

		stacknum += 1

plt.figure()
plt.xlabel('epsilon')
plt.ylabel('Structure similarity')
plt.plot([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], np.average(ssim, axis=0))
plt.show()
# plt.