# hwnet
Representation for Handwritten Word Images

## Dataset
To download synthetic data and its annotation file, navigate to hwnet/iiit-hws/ and refer the README.md

## Installation
The code is built using pytorch library. Following are the necessary packages to be installed:
+ Python 2.7
+ numpy, sklearn
+ opencv 2.4
+ PIL
+ Pytorch 0.2 and torchvision
+ \<optional but desired\> CUDA 8.0 and CUDNN

### Computing image features for a new corpus of word images.
#### Pre-requisite data <default-locations>
+ Image folder \<wordImages/\>: Containing word images for testing.
+ Test Annotation File \<ann/test_ann.txt\>: The file is given as the input for feature extraction. It has the following syntax in each line corresponding to each word image/string:<br>
```<word-img1-path><space><dummyString><space><dummyInt><space>1```<br>
```<word-img2-path><space><dummyString><space><dummyInt><space>1```<br>
...<br>
+ Pretrained Model \<pretrained/\>: Please download the pretrained model file for [IAM dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) from below URL and store it in pretrained/iam-model.t7 location<br>
  + IAM Model: http://ocr.iiit.ac.in/data/models/hwnet/pytorch/IAM/iam-model.t7
  + For other datasets, please refer to the location: hwnet/models/

```
cd pytorch
python hwnet-feat.py --annFile ../ann/test_ann.txt --pretrained_file pretrained/iam-model.t7 --img_folder ../wordImages/ --testAug --exp_dir output/ --exp_id iam-test-0
```
The above code will compute features and save it numpy matrices in location ```output/models/iam-test-0/```. Here feats.npy will contain featues for word images in the order provided in annotation file. The dimension of the matrix would be Nx2048. Here 'N' is the number of word images and 2048 is the feature dimension for the current trained model.
  
Arguments for running above code:
+ annFile: test annotation file
+ pretrained_file: pretrained model file.
+ img_folder: folder location containing word images
+ testAug: test time augmentation flag. If used, will compute features at multiple word image sizes (32, 48, 64) and combine the features using max pooling. 
+ exp_dir: folder where output files will be stored
+ exp_id: sub folder for the current experiment.

There are other arguments in the code. Please keep the default setting for current purpose.

### Evaluation of Query-By-Image Word Spotting
#### Pre-requisite data <default-locations>
+ Test Annotation File \<ann/test_ann.txt\>: The file syntax is same as described above, except the <dummyString> field should contain the actual unicode string/ground truth text associated with the corresponding word image. 
+ Query File \<ann/test_query.txt>\: Query file contains the query indexes. Each index is an integer value which points to the query image from annFile. The syntax of this file is:
```1```<br>
```3```<br>
...<br>
Here word images at location 1,3,... from file test_ann.txt will be used for querying.

```
cd pytorch
python eval.py --exp_dir output/ --exp_id iam-test-0 --annFile ../ann/test_ann.txt --query_file ../ann/test_query.txt
```
The above code will compute average precision scores for each query and finally dump the mean average precision (mAP) for the entire dataset.

Arguments for running above code:
+ annFile: test annotation file
+ query_file: File containing the query indexes. Each index is an integer value which points to the query image from annFile. Note that in general, stopwords are not used for querying. Therefore the query file should contain all indexes (one index in each line) without including stopwords.
+ exp_dir: folder where output files will be stored.
+ exp_id: sub folder for the current experiment.
+ printFlag: Use this flag to print the retrival list.

There are other arguments in the code. Please keep the default setting for current purpose.

Sample wordImages, annotation file and the query file is kept in its respective folder location.

## Citation
If you are using the dataset, please cite the below arxiv paper:-
- Praveen Krishnan and C.V. Jawahar, Generating Synthetic Data for Text Recognition, arXiv preprint arXiv:1608.04224, 2016.

If you are comparing our method for word spotting/recognition, please cite the below relevant papers:-
- Praveen Krishnan and C.V. Jawahar, HWNet v2: an efficient word image representation for handwritten documents, IJDAR 2019.
- Praveen Krishnan and C.V. Jawahar, Matching Handwritten Document Images, ECCV 2016

## Contact
Incase of any doubts, please contact the author using below details:-<br>
Author Name: Praveen Krishnan<br>
Author Email: praveen.krishnan@research.iiit.ac.in<br>
