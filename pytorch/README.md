# hwnet
Representation for Handwritten Word Images

## Installation
The code is built using pytorch library. Following are the necessary packages to be installed:
+ Python 2.7
+ numpy, sklearn, nltk
+ opencv 3.2
+ PIL
+ Pytorch 0.3.1 and torchvision
+ \<optional but desired\> CUDA 8.0 and CUDNN

### Pre-requisite data <default-locations>
+ Image folder \<wordImages/\>: Containing word images for testing.
+ Test Annotation File \<ann/test_ann.txt\>: The file is given as the input for feature extraction. It has the following syntax in each line corresponding to each word image/string:<br>
```<word-img1-path><space><text1-string><space><dummyInt><space>1```<br>
```<word-img2-path><space><text2-string><space><dummyInt><space>1```<br>
...<br>
+ Pretrained Model \<pretrained/\>: Please download the pretrained model file for [IAM dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) from below URL and store it in pretrained/iam-model.t7 location<br>
  + IAM Model: http://ocr.iiit.ac.in/data/models/hwnet/pytorch/IAM/iam-model.t7

    
### Computing image and text features for a new corpus of word images.
```
python hwnet-feat.py --annFile ann/test_ann.txt --pretrained_file pretrained/iam-model.t7 --img_folder wordImages/ --testAug --exp_dir output/ --exp_id iam-test-0
```
The above code will compute features and save it numpy matrices in location ```output/models/iam-test-0/```. Here feats.npy will contain featues for word images in the order provided in annotation file. The dimension of the matrix would be Nx2048. Here 'N' is the number of word images and 2048 is the feature dimension for the current trained model.
  
Arguments for running above code:
+ test_vocab_file: test annotation file
+ pretrained_file: pretrained model file.
+ img_folder: folder location containing word images
+ testAug: test time augmentation flag. If used, will compute features at multiple word image sizes (32, 48, 64) and combine the features using max pooling. 
+ exp_dir: folder where output files will be stored
+ exp_id: sub folder for the current experiment.

There are other arguments in the code. Please keep the default setting for current purpose.
