# AMC-SSDA
Adaptive Multi Column Sparse Stacked Denoising Autoencoder 

# Collaborator: 
Prayushi Mathur, Vraj Patel

This repository contains the files related to Denoising of images containing multiple types of noise using AMC-SSDA. The dataset used here is Mnist.

# Introduction
-Denoising the images is a very important task when working with Computer Vision and Image Processing.
-In some applications, the noisy images are taken as input where there is chance to lose some important features of that image.
-There are various types of noise present at a same time in the training images and removing them all is a difficult task.
-Here, we introduce a technique called Adaptive Multi-Column Stacked Sparse Denoising Autoencoder (AMC-SSDA) which is use dto remove all the noise at the same time from the noisy images.

-There are mainly 3 parts in this technique:
1) Autoencoder
2) Qudratic program (QP)
3) Radial Basis Function (RBF)

1)  The autoencoder used here is a sparsed denoising autoencoder which helps to denoise a single type of noise. So we have to use the same number of SDAs as the number of type of noise present in the noisy image.
-Every autoencoder trained on different type of noise remove from images.
-During testing when an image is given as input to the AMC-SSDA, it will go to all these SSDAs and generate an output as per their training individually.

2) QP is used for getting an optimal solution to generate a single image from all those images which we have got from each SSDA.
-This part gives the optimum weights to each output generated from autoencoder according to target image during training.

3) During testing we do not have target images so that we cannot use QP here. Therefore, we have to use trained RBF which takes latent vector as input and optimal weight vector (which we got from above QP part) as output. 

# IMAGE

# Introduction of Files Contained in this repository
Display_Digits.py - This file is used to convert image dimension from (784,1) to (28,28) and to display those images.
PSNR.py - This file is used to find PSNR values of the generated outputs with respect to actual output. PSNR value tells the quality of a generated denoised image with respect to actual denoise image.
FeedForward_Sparse_AE.py - This file contains the code to create a sparse Autoencoder in tensorflow.
keras_AE.py - This file is used to create an autoencoder in keras.
SSDA.py - This file is used to create SSDA(stacked sparsed de-noising autoncoder).
main.py - This file contains the final code importing every file from above to generate final output.

# Disclaimer
- Here, we provide weights for all SSDA in form of json file.
- In the above code, we have shown this technique for mnist dataset and RBF which get trained for a single type noisy image. This can be done for all types of images.
-This technique can also be used for different types of noise as well as different variations of each noise.

# Conclusion
- This technique is working very efficiently when the image contains multiple type of noise with multiple complication of each noise present in input images.

- We have also implemented this technique for medical 2D brain images which contained multiple type of noisy images along with different variations of each noise.
