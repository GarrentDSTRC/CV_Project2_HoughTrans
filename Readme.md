

```csharp
pip install -r requirements.txt
```
### 1. Hough_Transform.py

This file has the code for detecting circles in a given image using Hough Transform.
The Radius range original_image_path, and Threshold can be changed and adjusted as per need in order to improve the performance of the program. The default setting is "example1" , 50-140 , 250. The optimal parameter of "example2"  is 100-150, 280.

### 2. Test.py

After drawing, the program will save the hough space as acc_array.npy. 

![alt](https://github.com/GarrentDSTRC/CV_Project2_HoughTrans/blob/master/Readme_md_files/image.png)  
Then, Test.py is used to combine the different acc_array.npy responding to different ranges of radius and test a proper threshold.
But those .npy files are too large, I have to delete them before handling them.
The results with different parameters are named as "Detected_circle_HT-example*-r*_*-thesholds *"
