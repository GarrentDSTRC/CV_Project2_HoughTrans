from __future__ import division
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

original_image = cv2.imread('example2.png',1)


output = original_image.copy()

height,width,length = original_image.shape
radii = 200

#acc_array = np.zeros(((height,width,radii)))

filter3D = np.zeros((30,30,radii))
filter3D[:,:,:]=1

start_time = time.time()

#combine two saved array
acc_array =np.load('acc_array1-50-100.npy')

if acc_array.shape[2]<200:
    ac=np.zeros(acc_array.shape)
    acc_array=np.concatenate((ac,acc_array),axis =2)

acc_array2=np.load('acc_array1-100-140.npy')

acc_array=acc_array+acc_array2



i = 0
j = 0
while (i < height - 30):
    while (j < width - 30):
        filter3D = acc_array[i:i + 30, j:j + 30, :] * filter3D
        max_pt = np.where(filter3D == filter3D.max())
        a = max_pt[0]
        b = max_pt[1]
        c = max_pt[2]
        b = b + j
        a = a + i
        if (filter3D.max() > 245):
            for o in range (len(a)):
                ao=a[o]
                bo=b[o]
                co=c[o]
                cv2.circle(output, (bo, ao), co, (0, 255, 0), 2)
        j = j + 30
        filter3D[:, :, :] = 1
    j = 0
    i = i + 30
    print(i)

cv2.imshow('Detected circle', output)
end_time = time.time()
time_taken = end_time - start_time
print('Time taken for execution', time_taken)

cv2.waitKey(0)
cv2.imwrite('Detected_circle_HT.png', output)
cv2.destroyAllWindows()