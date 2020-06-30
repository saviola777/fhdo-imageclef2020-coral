import os
import logging
import sys
import random
import math
import re
import time
import numpy as np
import csv
import glob
import progressbar
import cv2

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

RUN_NAME = sys.argv[1]

# Directories
MRCNN_DIR = os.path.abspath(
    "/path/to/Mask_RCNN")
ROOT_DIR = os.path.abspath("/path/to/models")
PRETRAINED_MODEL_DIR = os.path.abspath(
    "/path/to/pretrained/models")
DATA_DIR = os.path.abspath("/path/to/dataset/dir")
# Customization checklist:
# - dataset (mask correction?)
# - classes
# - iamge size & anchor scales
FILE_SUFFIX = ""
ANNOTATION_CSV_POLYGONS = '/path/to/annotations_train_2020_polygons-corrected.cvs.txt'
ANNOTATION_CSV_BOXES = '/path/to/annotations_train_2020_boxes-corrected.cvs.txt'
ANNOTATION_CSV_TEST = '/path/to/test_files.csv'

###############################
IMAGE_SUBDIR = "train_ibla"
IMAGE_SUBDIR_TEST = "test_ibla"
IMAGE_SUFFIX = "_IBLA.jpg"
##############################


#################
SEGMENTATION = True
LARGER_IMAGES = True
IMAGE_SIZE = 1536 if LARGER_IMAGES else 1024
################

MEAN_PIXEL_IBLA = np.array([82.69938608,  90.13531994, 90.4803112])
MEAN_PIXEL_REDUCED = np.array([82.15477202, 89.65388097, 90.01923429])
MEAN_PIXEL_RAYLEIGH = np.array([103.07894043, 100.80574327, 99.79139372])
MEAN_PIXEL_ORIGINAL = np.array([64.07844228, 108.6276674, 100.78435131])
MEAN_PIXEL_TEST_ORIGINAL = [68.73740876, 110.24671141,  91.16937015]
MEAN_PIXEL_TEST_IBLA = [72.76736444, 86.93823197, 79.62567819]
MEAN_PIXEL_TEST_RAYLEIGH = [99.36580744, 100.04721066,  98.12982314]
MEAN_PIXEL_TEST_REDUCED = [72.25588446, 86.39932899, 79.15395045]

#####################################
MEAN_PIXEL_CURRENT = MEAN_PIXEL_IBLA
MEAN_PIXEL_TEST = MEAN_PIXEL_TEST_IBLA
#####################################

CLASSES = CLASSES_FULL = ["c_soft_coral", "c_sponge", "c_hard_coral_boulder", "c_hard_coral_branching",
           "c_hard_coral_encrusting", "c_hard_coral_submassive",
           "c_hard_coral_table", "c_hard_coral_foliose",
           "c_hard_coral_mushroom", "c_algae_macro_or_leaves",
           "c_soft_coral_gorgonian", "c_sponge_barrel",
           "c_fire_coral_millepora"]

#############################
#CLASSES = ["c_soft_coral", "c_sponge", "c_hard_coral_boulder", "c_hard_coral_branching", "c_hard_coral_encrusting"]
CROSS_VALIDATION = True
TRAIN_INFERENCE = True
############################

ORIGINAL_IMAGE_WIDTH = 1536
ORIGINAL_IMAGE_HEIGHT = 1536
MASK_CORRECTION = False
ANNOTATION_CSV_CURRENT = ANNOTATION_CSV_POLYGONS if SEGMENTATION else ANNOTATION_CSV_BOXES

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, RUN_NAME)
os.makedirs(MODEL_DIR, exist_ok=True)

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(PRETRAINED_MODEL_DIR, "mask_rcnn_coco.h5")

# Make sure TF does not print confusing warnings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
print(tf.__version__)
print(tf.__file__)
print(tf.keras.__version__)
print(tf.config.list_physical_devices('GPU'))

#from tensorflow.keras.mixed_precision import experimental as mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)
#print('Compute dtype: %s' % policy.compute_dtype)
#print('Variable dtype: %s' % policy.variable_dtype)

# Import Mask RCNN
sys.path.append(MRCNN_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from tensorflow.keras.callbacks import Callback

import asyncio
import concurrent

asyncio.get_event_loop().set_default_executor(
    concurrent.futures.ThreadPoolExecutor(max_workers=8))


tf.keras.backend.clear_session()


print("Imports done")
print("Directories:")
print("Root directory:", ROOT_DIR)
print("Model directory:", MODEL_DIR)
print("Pre-trained model directory:", PRETRAINED_MODEL_DIR)
print("M-RCNN directory:", MRCNN_DIR)
print("Datasets directory:", DATA_DIR)

random_indices = [371, 176, 23, 425, 17, 195, 363, 119, 7, 253, 249, 356, 327, 134, 236, 57, 15, 430, 72, 307, 404, 163, 239, 389, 388, 48, 152, 300, 355, 25, 110, 296, 240, 14, 154, 217, 112, 91, 198, 115, 88, 357, 429, 169, 319, 263, 342, 438, 213, 92, 18, 11, 27, 312, 108, 338, 422, 257, 299, 164, 274, 415, 387, 308, 265, 383, 376, 416, 366, 182, 238, 84, 328, 435, 158, 329, 207, 335, 340, 89, 62, 83, 2, 302, 373, 130, 144, 341, 96, 71, 289, 202, 395, 350, 352, 305, 172, 24, 354, 343, 232, 40, 390, 132, 411, 375, 315, 334, 227, 4, 103, 303, 433, 122, 184, 36, 81, 5, 120, 28, 372, 73, 349, 49, 146, 43, 306, 147, 77, 400, 237, 316, 251, 243, 74, 212, 109, 171, 87, 52, 199, 21, 39, 420, 145, 293, 222, 183, 410, 418, 111, 173, 194, 16, 98, 93, 351, 260, 242, 317, 258, 426, 407, 170, 378, 181, 235, 248, 142, 177, 106, 185, 330, 336, 437, 436, 159, 277, 124, 69, 225, 33, 428, 353, 401, 282, 374, 204, 230, 197, 187, 51, 434, 100, 246, 311, 284, 318, 218, 245, 150, 423, 167, 203, 295, 140, 364, 377, 424, 297, 368, 201, 234, 304, 259, 37, 392, 128, 206, 333, 161, 215, 136, 55, 406, 432, 226, 131, 97, 382, 244, 46, 264, 42, 348, 309, 41, 80, 254, 129, 391, 178, 439, 59, 30, 275, 1, 283, 288, 229, 151, 205, 324, 313, 380, 34, 90, 186, 104, 408, 272, 76, 19, 105, 12, 322, 233, 143, 361, 99, 135, 56, 231, 379, 86, 270, 113, 45, 301, 267, 346, 61, 162, 127, 6, 190, 114, 3, 320, 339, 398, 393, 116, 431, 256, 345, 386, 405, 326, 412, 250, 362, 220, 153, 189, 280, 191, 427, 290, 117, 44, 219, 276, 137, 141, 174, 58, 358, 160, 94, 440, 118, 68, 271, 138, 396, 298, 67, 273, 223, 228, 60, 314, 278, 337, 384, 381, 399, 149, 325, 367, 47, 70, 22, 193, 281, 168, 126, 26, 359, 102, 50, 139, 148, 155, 196, 347, 294, 95, 209, 385, 403, 20, 365, 421, 417, 285, 332, 208, 101, 157, 64, 269, 419, 66, 344, 287, 125, 397, 266, 75, 179, 188, 413, 321, 370, 247, 214, 8, 310, 192, 29, 107, 13, 241, 402, 123, 286, 369, 291, 255, 166, 175, 82, 409, 268, 31, 323, 32, 9, 156, 414, 53, 224, 210, 85, 331, 79, 261, 63, 262, 133, 121, 252, 38, 211, 279, 180, 394, 35, 54, 165, 10, 65, 216, 292, 360, 200, 78, 221]


class CoralInferenceConfig(Config):
    """Configuration for training on the coral dataset.
    Derives from the base Config class and overrides values specific
    to the coral dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coral"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(CLASSES)

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    #RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    RPN_ANCHOR_SCALES = (48,  96, 192, 384, 768) if LARGER_IMAGES else (32, 64, 128, 256, 512)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    LEARNING_RATE = 0.01

    LEARNING_MOMENTUM = 0.9

    WEIGHT_DECAY = 0.0001

    DETECTION_MIN_CONFIDENCE = 0.6

    MEAN_PIXEL = MEAN_PIXEL_CURRENT

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 44

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 11

    USE_MINI_MASK = False


##
# Start dataset
##

def map_mask(xmin, xmax, ymin, ymax):
    return [
        max(0, min(math.floor(int(xmin) / 4032 * ORIGINAL_IMAGE_WIDTH), ORIGINAL_IMAGE_WIDTH) - 1),
        max(0, min(math.ceil(int(xmax) / 4032 * ORIGINAL_IMAGE_WIDTH), ORIGINAL_IMAGE_WIDTH) - 1),
        max(0, min(math.floor(int(ymin) / 3024 * ORIGINAL_IMAGE_HEIGHT), ORIGINAL_IMAGE_HEIGHT) - 1),
        max(0, min(math.ceil(int(ymax) / 3024 * ORIGINAL_IMAGE_HEIGHT), ORIGINAL_IMAGE_HEIGHT) - 1),
    ]

#def remap_mask(xmin, xmax, ymin, ymax):
#    return [
#        math.ceil(int(xmin + 1) * 4032 / 1536),
#        math.floor(int(xmax + 1) * 4032 / 1536),
#        math.ceil(int(ymin + 1) * 3024 / 1536),
#        math.floor(int(ymax + 1) * 3024 / 1536)
#    ]


def remap_mask(xmin, xmax, ymin, ymax):
    return [
        math.ceil(int(xmin + 1) * 4032 / IMAGE_SIZE),
        math.floor(int(xmax + 1) * 4032 / IMAGE_SIZE),
        math.ceil(int(ymin + 1) * 3024 / IMAGE_SIZE),
        math.floor(int(ymax + 1) * 3024 / IMAGE_SIZE)
    ]

def map_polygon(points):
    if len(points) % 2 != 0:
        raise ValueError("Number of points must be even")

    for i in range(0, len(points), 2):
        points[i] = max(0, min(math.floor(int(points[i]) / 4032 * ORIGINAL_IMAGE_WIDTH), ORIGINAL_IMAGE_WIDTH) - 1)
        points[i+1] = max(0, min(math.floor(int(points[i+1]) / 3024 * ORIGINAL_IMAGE_HEIGHT), ORIGINAL_IMAGE_HEIGHT) - 1)

    return points


def mask_to_polygon(mask, roi):
    if np.sum(mask[roi[0]:roi[2], roi[1]:roi[3]]) < 100:
        return []

    contours, hierarchy = cv2.findContours(
        mask[roi[0]:roi[2], roi[1]:roi[3]].astype(np.uint8)*255,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS,
        offset=(roi[1], roi[0]))

    polygon = []
    if len(contours) == 0:
        print(f"No polygon found. np.max(mask): {np.max(mask)}. ROI: {roi}")
        return polygon

    lengths = [cv2.contourArea(x) for x in contours]
    contour = contours[lengths.index(np.max(lengths))]

    if len(contour) < 3:
        print(f"Less than 3 points found: {contours[0]} ({roi}), np.sum(mask): {np.sum(mask)}")

    seen_points = []
    for i in range(len(contour)):
        x = contour[i][0][0]
        y = contour[i][0][1]
        p = (x, y)
        h = hash(p)
        if h not in seen_points:
            polygon.append(round(int(contour[i][0][0] + 1) * 4032 / IMAGE_SIZE))
            polygon.append(round(int(contour[i][0][1] + 1) * 3024 / IMAGE_SIZE))
            seen_points.append(p)
            seen_points.append(hash((x+1, y)))
            seen_points.append(hash((x, y+1)))
            seen_points.append(hash((x-1, y)))
            seen_points.append(hash((x, y-1)))

    while len(polygon) > 1000:
        remove_elements = []
        for i in range(0, (len(polygon) // 2) - 1, 2):
            remove_elements.append(i * 2)
            remove_elements.append(i * 2 + 1)
        remove_elements.sort(reverse=True)
        for e in remove_elements:
            polygon.pop(e)

    return polygon


class CoralDataset(utils.Dataset):
    """Generates the bone age dataset.
    """

    def add_coral_image(self, image_indices, i, annotations, image_dir,
                        current_filename):
        if len(annotations) > 0 and (len(image_indices) == 0 or i in image_indices):
            self.add_image(
                "coral", image_id=i,
                path=os.path.join(image_dir, current_filename + IMAGE_SUFFIX),
                # width=4032,
                # height=3024,
                width=ORIGINAL_IMAGE_WIDTH,
                height=ORIGINAL_IMAGE_HEIGHT,
                annotations=annotations)

    def load_coral(self, dataset_dir, annotation_file, image_indices):
        """Load the requested subset of the bone age dataset.
        """
        # Add classes
        for i in range(len(CLASSES)):
            self.add_class("coral", i+1, CLASSES[i])

        # Add images from CSV
        with open(annotation_file, 'r') as input_file:

            reader = csv.reader(input_file, delimiter=' ')

            # Skip header
            # header = next(reader)

            image_dir = dataset_dir
            current_filename, current_width, current_height = [None, None, None]

            annotations = []
            i = 1
            for row in reader:
                filename, index, class_name, confidence, xmin, ymax, xmax, ymin = row

                xmin, xmax, ymin, ymax = map_mask(xmin, xmax, ymin, ymax)
                #width, height = [xmax - xmin, ymax - ymin]

                if filename != current_filename and current_filename != None:
                    # annotations.sort(key=lambda x: x[0] + str(x[1]))
                    # annotations = list(map(incremental_class_names, annotations))

                    self.add_coral_image(image_indices, i, annotations,
                                         image_dir, current_filename)

                    i += 1
                    annotations = []
                    # for c in class_name_counters:
                    # class_name_counters[c] = 0

                if class_name not in CLASSES:
                    continue

                current_filename = filename
                #current_width = width
                #current_height = height
                annotations.append([class_name, xmin, ymin, xmax, ymax])
                # annotations.append(["o", xmin, ymin, xmax, ymax])

            self.add_coral_image(image_indices, i, annotations, image_dir,
                                 current_filename)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "coral":
            return "https://www.imageclef.org/2020/coral"
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for instances in the given image.
        """
        instance_masks = []
        class_ids = []
        image_info = self.image_info[image_id]
        annotations = image_info["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.class_names.index(annotation[0])
            if class_id:
                m = np.full((image_info["height"], image_info["width"]),
                            fill_value=False)
                m[annotation[2]:annotation[4], annotation[1]:annotation[3]] = True

                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() == False:
                    # print("Skipped one!")
                    continue

                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        masks = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)
        return masks, class_ids


class CoralFullDataset(CoralDataset):
    """Generates the bone age dataset.
    """

    def load_coral(self, dataset_dir, annotation_file, image_indices):
        """Load the requested subset of the bone age dataset.
        """
        # Add classes
        for i in range(len(CLASSES_FULL)):
            self.add_class("coral", i+1, CLASSES_FULL[i])

        # Add images from CSV
        with open(annotation_file, 'r') as input_file:

            reader = csv.reader(input_file, delimiter=' ')

            # Skip header
            # header = next(reader)

            image_dir = dataset_dir
            current_filename, current_width, current_height = [None, None, None]

            annotations = []
            i = 1
            for row in reader:
                filename, index, class_name, confidence, xmin, ymax, xmax, ymin = row

                xmin, xmax, ymin, ymax = map_mask(xmin, xmax, ymin, ymax)
                #width, height = [xmax - xmin, ymax - ymin]

                if filename != current_filename and current_filename != None:
                    # annotations.sort(key=lambda x: x[0] + str(x[1]))
                    # annotations = list(map(incremental_class_names, annotations))

                    self.add_coral_image(image_indices, i, annotations,
                                         image_dir, current_filename)

                    i += 1
                    annotations = []
                    # for c in class_name_counters:
                    # class_name_counters[c] = 0

                if class_name not in CLASSES_FULL:
                    continue

                current_filename = filename
                #current_width = width
                #current_height = height
                annotations.append([class_name, xmin, ymin, xmax, ymax])
                # annotations.append(["o", xmin, ymin, xmax, ymax])

            self.add_coral_image(image_indices, i, annotations, image_dir,
                                 current_filename)


class CoralSegmentationDataset(CoralDataset):
    """Generates the coral dataset.
    """

    def load_coral(self, dataset_dir, annotation_file, image_indices):
        """Load the requested subset of the bone age dataset.
        """
        # Add classes
        for i in range(len(CLASSES)):
            self.add_class("coral", i+1, CLASSES[i])

        # Add images from CSV
        with open(annotation_file, 'r') as input_file:

            reader = csv.reader(input_file, delimiter=' ')

            # Skip header
            # header = next(reader)

            image_dir = dataset_dir
            current_filename, current_width, current_height = [None, None, None]

            annotations = []
            i = 1
            for row in reader:
                filename, index, class_name, confidence = row[:4]
                polygon = map_polygon(row[4:])

                if filename != current_filename and current_filename != None:
                    # annotations.sort(key=lambda x: x[0] + str(x[1]))
                    # annotations = list(map(incremental_class_names, annotations))

                    self.add_coral_image(image_indices, i, annotations,
                                         image_dir, current_filename)

                    i += 1
                    annotations = []
                    # for c in class_name_counters:
                    # class_name_counters[c] = 0

                if class_name not in CLASSES:
                    continue

                current_filename = filename
                annotations.append([class_name, polygon])


            self.add_coral_image(image_indices, i, annotations, image_dir,
                                 current_filename)

    def load_mask(self, image_id):
        """Generate instance masks for instances in the given image.
        """
        instance_masks = []
        class_ids = []
        image_info = self.image_info[image_id]
        annotations = image_info["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.class_names.index(annotation[0])
            polygon_points = np.reshape(annotation[1], (1, -1, 2)).astype(np.int32)
            if class_id:
                m = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
                #cv2.fillPoly(m, polygon_points, 1)
                cv2.drawContours(m, polygon_points, -1, (1), -1)

                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    # print("Skipped one!")
                    continue

                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        #masks = np.reshape(np.stack(instance_masks, axis=2).astype(np.bool), (
        #1, image_info["height"], image_info["width"], len(class_ids)))[0]
        try:
            masks = np.stack(instance_masks, axis=2)
        except ValueError as e:
            print(f"Found {len(annotations[1])} polygon point for image {self.image_info[image_id]['path']}")
            raise e

        class_ids = np.array(class_ids, dtype=np.int32)
        return masks, class_ids


class CoralSegmentationFullDataset(CoralSegmentationDataset):
    """Generates the coral dataset.
    """

    def load_coral(self, dataset_dir, annotation_file, image_indices):
        """Load the requested subset of the bone age dataset.
        """
        # Add classes
        for i in range(len(CLASSES_FULL)):
            self.add_class("coral", i+1, CLASSES_FULL[i])

        # Add images from CSV
        with open(annotation_file, 'r') as input_file:

            reader = csv.reader(input_file, delimiter=' ')

            # Skip header
            # header = next(reader)

            image_dir = dataset_dir
            current_filename, current_width, current_height = [None, None, None]

            annotations = []
            i = 1
            for row in reader:
                filename, index, class_name, confidence = row[:4]
                polygon = map_polygon(row[4:])

                if filename != current_filename and current_filename != None:
                    # annotations.sort(key=lambda x: x[0] + str(x[1]))
                    # annotations = list(map(incremental_class_names, annotations))

                    self.add_coral_image(image_indices, i, annotations,
                                         image_dir, current_filename)

                    i += 1
                    annotations = []
                    # for c in class_name_counters:
                    # class_name_counters[c] = 0

                if class_name not in CLASSES_FULL:
                    continue

                current_filename = filename
                annotations.append([class_name, polygon])


            self.add_coral_image(image_indices, i, annotations, image_dir,
                                 current_filename)


class CoralTestDataset(CoralDataset):
    """Generates the bone age dataset.
        """
    def add_coral_image(self, image_indices, i, annotations, image_dir,
                        current_filename):
        if len(annotations) > 0 and (
                len(image_indices) == 0 or i in image_indices):
            self.add_image(
                "coral", image_id=i,
                path=os.path.join(image_dir, current_filename + IMAGE_SUFFIX),
                # width=4032,
                # height=3024,
                width=ORIGINAL_IMAGE_WIDTH,
                height=ORIGINAL_IMAGE_HEIGHT,
                annotations=annotations)

    def load_coral(self, dataset_dir, annotation_file, image_indices=[]):
        """Load the requested subset of the bone age dataset.
        """
        # Add classes
        for i in range(len(CLASSES)):
            self.add_class("coral", i + 1, CLASSES[i])

        # Add images from CSV
        with open(annotation_file, 'r') as input_file:
            reader = csv.reader(input_file, delimiter=',')

            # Skip header
            next(reader)

            image_dir = dataset_dir

            annotations = [0]
            i = 1
            for row in reader:
                current_filename = row[0]

                self.add_coral_image(image_indices, i, annotations,
                                     image_dir, current_filename)
                i += 1

    def load_mask(self, image_id):
        return np.zeros((self.image_info[image_id]["height"], self.image_info[image_id]["width"], 1),
                     dtype=np.uint8), np.array([0], dtype=np.int32)


def create_model(mode, config, init_with, i):
    model_dir = os.path.join(MODEL_DIR, f"{config.NAME}{i}")
    os.makedirs(model_dir, exist_ok=True)
    # Create model in training mode
    model = modellib.MaskRCNN(mode=mode, config=config,
        model_dir=model_dir)

    # Which weights to start with?
    #init_with = "last"  # imagenet, coco, manual or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
    else:
        model.load_weights(init_with, by_name=True)

    print("Model loaded")
    return model


def get_all_indices():
    return random_indices


def get_indices_train(i):
    return list(set(random_indices) - set(get_indices_val(i)))


def get_indices_val(i):
    return random_indices[i * 88: i * 88 + 88]


def collect_dataset_predictions(dataset, model, split, data_split):
    global progress

    batches = math.ceil(
            len(dataset.image_ids) / config_inference.IMAGES_PER_GPU)

    rows_boxes = []
    rows_polygons = []
    APs = []
    for batch in range(batches):
        #print(f"Starting batch {batch}/{batches}")
        image_ids = dataset.image_ids[batch * config_inference.IMAGES_PER_GPU:(batch + 1) * config_inference.IMAGES_PER_GPU]
        images = []
        image_paths = []

        for i in image_ids:
            # original_image, image_meta, gt_class_id, gt_bbox, gt_mask
            images.append(
                modellib.load_image_gt(dataset, config_inference, i))
            image_paths.append(dataset.image_info[i]["path"])

        # (y1, x1, y2, x2, class_id, score)
        results = model.detect(list(x[0] for x in images))

        for i in range(len(results)):
            r = results[i]

            if MASK_CORRECTION:
                for j in range(len(r["rois"])):
                    # ymin:ymax, xmin:xmax
                    r["masks"][j][r["rois"][j][0]:r["rois"][j][2], r["rois"][j][1]:r["rois"][j][3]] = 1

            # Compute AP
            AP, precisions, recalls, overlaps = \
                utils.compute_ap(images[i][3], images[i][2], images[i][4],
                                 r["rois"], r["class_ids"], r["scores"],
                                 r['masks'],
                                 iou_threshold=IOU_THRESHOLD)

            APs.append(AP)

            for j in range(len(r["rois"])):
                xmin, xmax, ymin, ymax = remap_mask(r["rois"][j][1],
                           r["rois"][j][3],
                           r["rois"][j][0],
                           r["rois"][j][2])
                rows_boxes.append([
                    RUN_NAME,
                    split,
                    data_split,
                    best_models[split][0][1],
                    image_paths[i],
                    j,
                    dataset_train.class_names[r["class_ids"][j]],
                    r["scores"][j],
                    xmin,
                    ymax,
                    xmax,
                    ymin
                ])
                polygon_points = mask_to_polygon(r["masks"][:, :, j], r["rois"][j])
                #if len(polygon_points) <= 4 or len(polygon_points) % 2 != 0:
                    #polygon_points = roi_to_polygon(
                if len(polygon_points) > 4 and len(polygon_points) % 2 == 0:
                    rows_polygons.append([
                        RUN_NAME,
                        split,
                        data_split,
                        best_models[split][0][1],
                        image_paths[i],
                        j,
                        dataset_train.class_names[r["class_ids"][j]],
                        r["scores"][j],
                        *polygon_points
                    ])
                else:
                    print(f"{image_paths[i]}: Invalid polygon ({polygon_points} ({len(polygon_points)}) found for ROI {r['rois'][j]}, confidence {r['scores'][j]}, np.sum(mask): {np.sum(r['masks'][:, :, j])}")

        next(progress)

    return APs, rows_boxes, rows_polygons


##
# End configuration and classes
##

IOU_THRESHOLD = .5

# CUSTOMIZE: switch commenting when fixing existing evals
progress = progressbar.progressbar(range(1, (55*5 + 1) if CROSS_VALIDATION else (440//8 + 400//8 +1)))
#progress = progressbar.progressbar(range(1, 20 + 1))


def find_best_models(num_models):
    model_losses = [[], [], [], [], []]
    with open(os.path.join(MODEL_DIR, "losses.csv"), 'r') as input_csv:
        reader = csv.reader(input_csv, delimiter=',')
        next(reader)

        for row in reader:
            run_name, split, epoch, val_loss, train_loss = row[:5]
            #train_loss = row[11] + row[12] # MRCNN class + bbox loss
            #val_loss = row[7] + row[8]


            glob_search = os.path.join(MODEL_DIR, "coral" + split) +\
                          '/*/mask_rcnn_coral_0' + str(int(epoch) + 1).zfill(3)\
                          + '.h5'
            current_loss = float(train_loss) * .368 + float(val_loss) * .632
            try:
                model_losses[int(split)].append(
                    (current_loss, glob.glob(glob_search)[0]))
            except IndexError:
                #print(f"Warning: Didn't find model for glob {glob_search}")
                pass

    return list(sorted(x, key=lambda k: k[0])[:num_models] for x in model_losses)


config_inference = CoralInferenceConfig()

with open(os.path.join(MODEL_DIR, f"evaluation{FILE_SUFFIX}.csv"), 'w') as output_csv_boxes:
    writer_boxes = csv.writer(output_csv_boxes, delimiter=',')
    writer_boxes.writerow(
        ['run_name', 'split', 'model_path', 'mAP_val', 'mAP_sd_val',
         'mAP_train', 'mAP_sd_train', '.632_error', '.632_error_sd', 'epoch', 'epoch_sd'])

mAPs_train = []
mAPs_val = []
mAPs_sd_train = []
mAPs_sd_val = []
best_model_epoch = []
best_model_epoch_sd = []
dot_632_error = []
dot_632_error_sd = []
best_models = find_best_models(5)
for split in range(5):

    print(f"Start split {split}")

    # Training dataset
    dataset_train = CoralSegmentationDataset() if SEGMENTATION else CoralDataset()
    dataset_train.load_coral(os.path.join(DATA_DIR, IMAGE_SUBDIR),
                             os.path.join(DATA_DIR, ANNOTATION_CSV_CURRENT),
                            get_indices_train(split) if CROSS_VALIDATION else get_all_indices())
    dataset_train.prepare()
    print("Images in training set: " + str(len(dataset_train.image_info)))

    dataset_val = CoralSegmentationDataset() if SEGMENTATION else CoralDataset()
    dataset_val.load_coral(os.path.join(DATA_DIR, IMAGE_SUBDIR),
                             os.path.join(DATA_DIR, ANNOTATION_CSV_CURRENT),
                            get_indices_val(split) if CROSS_VALIDATION else get_all_indices())
    dataset_val.prepare()
    print("Images in validation set: " + str(len(dataset_val.image_info)))

    # Training dataset
    dataset_train_full = CoralSegmentationFullDataset() if SEGMENTATION else CoralFullDataset()
    dataset_train_full.load_coral(os.path.join(DATA_DIR, IMAGE_SUBDIR),
                                os.path.join(DATA_DIR, ANNOTATION_CSV_CURRENT),
                                  get_indices_train(split) if CROSS_VALIDATION else get_all_indices())
    dataset_train_full.prepare()
    print("Images in full training set: " + str(
        len(dataset_train_full.image_info)))

    if CROSS_VALIDATION:
        dataset_val_full = CoralSegmentationFullDataset() if SEGMENTATION else CoralFullDataset()
        dataset_val_full.load_coral(os.path.join(DATA_DIR, IMAGE_SUBDIR),
                                    os.path.join(DATA_DIR, ANNOTATION_CSV_CURRENT),
                                    get_indices_val(split))
    else:
        config_inference.MEAN_PIXEL = MEAN_PIXEL_TEST
        dataset_val_full = CoralTestDataset()
        dataset_val_full.load_coral(os.path.join(DATA_DIR, IMAGE_SUBDIR_TEST),
                                    os.path.join(DATA_DIR,
                                                 ANNOTATION_CSV_TEST))

    dataset_val_full.prepare()
    print("Images in full validation set: " + str(
        len(dataset_val_full.image_info)))

    model = create_model("inference", config_inference, list(x[1] for x in best_models[split][:5]), split)

    with open(os.path.join(MODEL_DIR, "coral" + str(split), f"predictions{FILE_SUFFIX}_boxes.csv"), 'w') as output_csv_boxes, open(os.path.join(MODEL_DIR, "coral" + str(split), f"predictions{FILE_SUFFIX}_polygons.csv"), 'w') as output_csv_polygons:

        writer_boxes = csv.writer(output_csv_boxes, delimiter=',')
        writer_boxes.writerow(
            ['run_name', 'cv_split', 'data_split', 'model_path', 'image_path', 'box_id',
             'class_name', 'confidence', 'x_min', 'y_max', 'x_max', 'y_min'])
        writer_polygons = csv.writer(output_csv_polygons, delimiter=',')
        writer_polygons.writerow(
            ['run_name', 'cv_split', 'data_split', 'model_path', 'image_path',
             'box_id',
             'class_name', 'confidence', 'x0', 'y0'])

        if TRAIN_INFERENCE:
            APs_train, rows_boxes, rows_polygons = collect_dataset_predictions(dataset_train_full, model,
                                                                split, "train")
            writer_boxes.writerows(rows_boxes)
            writer_polygons.writerows(rows_polygons)

        APs_val, rows_boxes, rows_polygons = collect_dataset_predictions(dataset_val_full, model,
                                                          split, "valid")
        writer_boxes.writerows(rows_boxes)
        writer_polygons.writerows(rows_polygons)

        if TRAIN_INFERENCE:
            mAPs_train.append(np.mean(APs_train))
            mAPs_val.append(np.mean(APs_val))
            mAPs_sd_train.append(np.std(APs_train))
            mAPs_sd_val.append(np.std(APs_val))

            dot_632_error_current = []
            best_model_epoch_current = []
            for loss, model_path in best_models[split]:
                best_model_epoch_current.append(int(model_path.split("/")[-1].split(".")[0].split("_")[-1]))
                dot_632_error_current.append(loss)

            dot_632_error.append(np.mean(dot_632_error_current))
            dot_632_error_sd.append(np.std(dot_632_error_current))
            best_model_epoch.append(np.mean(best_model_epoch_current))
            best_model_epoch_sd.append(np.std(best_model_epoch_current))


    if TRAIN_INFERENCE:
        with open(os.path.join(MODEL_DIR, f"evaluation{FILE_SUFFIX}.csv"), 'a') as output_csv_boxes:
            writer_boxes = csv.writer(output_csv_boxes, delimiter=',')
            writer_boxes.writerow([RUN_NAME, split, best_models[split][0][1], mAPs_val[-1],
                                   mAPs_sd_val[-1], mAPs_train[-1], mAPs_sd_train[-1], dot_632_error[-1], dot_632_error_sd[-1], best_model_epoch[-1], best_model_epoch_sd[-1]])


    tf.keras.backend.clear_session()

    if not CROSS_VALIDATION:
        break

if TRAIN_INFERENCE:
    with open(os.path.join(MODEL_DIR, f"evaluation{FILE_SUFFIX}.csv"), 'a') as output_csv_boxes:
        writer_boxes = csv.writer(output_csv_boxes, delimiter=',')
        writer_boxes.writerow(
            [RUN_NAME, "ALL", "", np.mean(mAPs_val), np.std(mAPs_val),
             np.mean(mAPs_train), np.std(mAPs_train), np.mean(dot_632_error),
             np.std(dot_632_error), np.mean(best_model_epoch),
             np.std(best_model_epoch)])
        writer_boxes.writerow([f"\\specialcell{{{round(np.mean(mAPs_train)*1000)/1000} \\\\ ({round(np.std(mAPs_train)*1000)/1000})}} & " +
                        f"\\specialcell{{{round(np.mean(mAPs_valid)*1000)/1000} \\\\ ({round(np.std(mAPs_valid)*1000)/1000})}} & " +
                        f"\\specialcell{{{round(np.mean(mAPs_valid)*1000)/1000} \\\\ ({round(np.std(mAPs_valid)*1000)/1000})}} & "
                        ])
