# FHDO ImageCLEF 2020 Coral Submission Code

This repository contains the training and evaluation code used for our [ImageCLEF 2020 Coral](https://www.imageclef.org/2020/coral) submission.

## Getting started

- clone this repository
- clone [Mask\_RCNN](https://github.com/DiffPro-ML/Mask_RCNN/)
- make sure the dependencies listed below are installed
- prepare the datasets (see below)
- adjust the paths beginning with `/path/to` in all scripts
- run `python3 prepare.py run_name` followed by `python3 train.py run_name`

## Dependencies

- OpenCV 4.2.0
- Tensorflow GPU 2.1
- numpy 1.18.2
- progressbar2 3.51.0

## Dataset pre-processing

- for IBLA, we used [this](https://github.com/wangyanckxx/Single-Underwater-Image-Enhancement-and-Color-Restoration/blob/master/Underwater%20Image%20Color%20Restoration/IBLA/main.py) script
- for Rayleigh, we used [this](https://github.com/wangyanckxx/Single-Underwater-Image-Enhancement-and-Color-Restoration/blob/master/Underwater%20Image%20Enhancement/RayleighDistribution/main.py) script on top of the IBLA images
- for color reduction, we used [octree\_color\_quantizer](https://github.com/delimitry/octree_color_quantizer)
