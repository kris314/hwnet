# IIIT-HWS Dataset

ReadMe for accessing IIIT-HWS dataset which comprises of handwritten synthetic word images.

## Download location
| Description  | Download Link  | File Size  |
|---|---|---|
| IIIT-HWS image corpus  | [IIIT-HWS](http://ocr.iiit.ac.in/data/dataset/iiit-hws/iiit-hws.tar.gz) | 32 GB  |
|  Ground truth files | [Ground Truth-10K](http://ocr.iiit.ac.in/data/dataset/iiit-hws/IIIT-HWS-10K.txt)  | 33 MB  |
|  Ground truth files | [Ground Truth-90K](http://ocr.iiit.ac.in/data/dataset/iiit-hws/IIIT-HWS-90K.txt)  | 294 MB  |

## Dataset Details
- Image Format: .png
- Image Size: 48x128 px
- #Word Images: 90M
- Vocabulary Size: 90K
- #Word Images/class: 100
- #Synthetic fonts used for rendering: ~750

## Image Data
- Extract iiit-hws.tar.gz file.
- Image Directory structure:<br>
    Images_90K_Normalized\/\<classId\>\/\<imgId\>.png<br>
    ..<br>

## Ground Truth
NOTE: In our ECCV paper, we have used a subset of IIIT-HWS dataset using only 10K vocabulary. The ground truth file for the same can obatined in IIIT-HWS-10K.txt file kept in the same directory. <br>
The syntax of the ground truth file is as follows:

```<img1-path><space><text1-string><space><dummyInt><space><train/test flag>```<br>
```<img2-path><space><text2-string><space><dummyInt><space><train/test flag>```<br>
...<br>
Here, in <train/test flag> = 0 for train and 1 for test.

## Citation
If you are using the dataset, please cite the below arxiv paper:-
- Praveen Krishnan and C.V. Jawahar, Generating Synthetic Data for Text Recognition, arXiv preprint arXiv:1608.04224, 2016.

If you are comparing our method for word spotting/recognition, please cite the below relevant papers:-
- Praveen Krishnan, Kartik Dutta and C.V. Jawahar, Deep Feature Embedding for Accurate Recognition and Retrieval of Handwritten Text, ICFHR 2016.
- Praveen Krishnan and C.V. Jawahar, Matching Handwritten Document Images, ECCV 2016

## Contact
Incase of any doubts, please contact the author using below details:-<br>
Author Name: Praveen Krishnan<br>
Author Email: praveen.krishnan@research.iiit.ac.in<br>
