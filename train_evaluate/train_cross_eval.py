import os
import logging
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import csv
import imgaug
import imgaug.augmenters as iaa
import progressbar
from scipy.interpolate import make_interp_spline, BSpline

RUN_NAME = sys.argv[1]

# Directories
MRCNN_DIR = os.path.abspath(
    "/path/to/Mask_RCNN")
ROOT_DIR = os.path.abspath("/path/to/models")
PRETRAINED_MODEL_DIR = os.path.abspath(
    "/path/to/pretrained/models")
DATA_DIR = os.path.abspath("/path/to/data/dir/")
# put these inside the DATA_DIR
ANNOTATION_CSV_POLYGONS = 'annotations_train_2020_polygons-corrected.cvs.txt'
ANNOTATION_CSV_BOXES = 'annotations_train_2020_boxes-corrected.cvs.txt'

#######################################
IMAGE_SUBDIR = "train_ibla"
IMAGE_SUFFIX = "_IBLA.jpg"
#IMAGE_SUFFIX = "_IBLA_RayleighDistribution.jpg"
AUGMENTATION = True
OVERSAMPLING = True
SEGMENTATION = True
LARGER_IMAGES = True
IMAGE_SIZE = 1536 if LARGER_IMAGES else 1024
BATCH_SIZE = 1 if LARGER_IMAGES else 2
#IMAGE_SIZE = 1024
#BATCH_SIZE = 2
MEAN_PIXEL_IBLA = np.array([82.69938608,  90.13531994, 90.4803112])
MEAN_PIXEL_REDUCED = np.array([82.15477202, 89.65388097, 90.01923429])
MEAN_PIXEL_RAYLEIGH = np.array([103.07894043, 100.80574327, 99.79139372])
MEAN_PIXEL_ORIGINAL = np.array([64.07844228, 108.6276674, 100.78435131])
MEAN_PIXEL_CURRENT = MEAN_PIXEL_IBLA
CLASSES = ["c_soft_coral", "c_sponge", "c_hard_coral_boulder", "c_hard_coral_branching",
           "c_hard_coral_encrusting", "c_hard_coral_submassive",
           "c_hard_coral_table", "c_hard_coral_foliose",
           "c_hard_coral_mushroom", "c_algae_macro_or_leaves",
           "c_soft_coral_gorgonian", "c_sponge_barrel",
           "c_fire_coral_millepora"]
#CLASSES = ["c_soft_coral", "c_sponge", "c_hard_coral_boulder", "c_hard_coral_branching", "c_hard_coral_encrusting"]
EPOCHS = 20
CROSS_VALIDATION = True
######################################
ANNOTATION_CSV = ANNOTATION_CSV_POLYGONS if SEGMENTATION else ANNOTATION_CSV_BOXES
STEPS_PER_EPOCH_TRAIN = 352/BATCH_SIZE if CROSS_VALIDATION else 440/BATCH_SIZE
STEPS_PER_EPOCH_TRAIN = 3*STEPS_PER_EPOCH_TRAIN if OVERSAMPLING else STEPS_PER_EPOCH_TRAIN
STEPS_PER_EPOCH_VALIDATION = 88/BATCH_SIZE if CROSS_VALIDATION else 440/BATCH_SIZE
LEARN_RATE_INITIAL = 0.002 if OVERSAMPLING else 0.005
#LEARN_RATE_INITIAL = 0.002 if OVERSAMPLING else 0.005
#LEARN_RATE_INITIAL = 0.002 if OVERSAMPLING else 0.001
#LEARN_RATE_ALL_LAYERS = 0.001
LEARN_RATE_ALL_LAYERS = 0.0005

ORIGINAL_IMAGE_WIDTH = 1536
ORIGINAL_IMAGE_HEIGHT = 1536

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

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.val_loss = []
        self.val_rpn_class_loss = []
        self.val_rpn_bbox_loss = []
        self.val_mrcnn_class_loss = []
        self.val_mrcnn_bbox_loss = []
        self.loss = []
        self.rpn_class_loss = []
        self.rpn_bbox_loss = []
        self.mrcnn_class_loss = []
        self.mrcnn_bbox_loss = []

    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.rpn_class_loss.append(logs.get('rpn_class_loss'))
        self.rpn_bbox_loss.append(logs.get('rpn_bbox_loss'))
        self.mrcnn_class_loss.append(logs.get('mrcnn_class_loss'))
        self.mrcnn_bbox_loss.append(logs.get('mrcnn_bbox_loss'))

    def on_epoch_end(self, epoch, logs={}):
        global split
        self.val_loss.append(logs.get('val_loss'))
        self.val_rpn_class_loss.append(logs.get('val_rpn_class_loss'))
        self.val_rpn_bbox_loss.append(logs.get('val_rpn_bbox_loss'))
        self.val_mrcnn_class_loss.append(logs.get('val_mrcnn_class_loss'))
        self.val_mrcnn_bbox_loss.append(logs.get('val_mrcnn_bbox_loss'))
        with open(os.path.join(MODEL_DIR, "losses.csv"), 'a') as output_csv:
            writer = csv.writer(output_csv, delimiter=',')
            writer.writerow([RUN_NAME, split, epoch,
                             self.val_loss[-1],
                             self.loss[-1],
                             self.val_rpn_class_loss[-1],
                             self.val_rpn_bbox_loss[-1],
                             self.val_mrcnn_class_loss[-1],
                             self.val_mrcnn_bbox_loss[-1],
                             self.rpn_class_loss[-1],
                             self.rpn_bbox_loss[-1],
                             self.mrcnn_class_loss[-1],
                             self.mrcnn_bbox_loss[-1]])


random_indices = [371, 176, 23, 425, 17, 195, 363, 119, 7, 253, 249, 356, 327, 134, 236, 57, 15, 430, 72, 307, 404, 163, 239, 389, 388, 48, 152, 300, 355, 25, 110, 296, 240, 14, 154, 217, 112, 91, 198, 115, 88, 357, 429, 169, 319, 263, 342, 438, 213, 92, 18, 11, 27, 312, 108, 338, 422, 257, 299, 164, 274, 415, 387, 308, 265, 383, 376, 416, 366, 182, 238, 84, 328, 435, 158, 329, 207, 335, 340, 89, 62, 83, 2, 302, 373, 130, 144, 341, 96, 71, 289, 202, 395, 350, 352, 305, 172, 24, 354, 343, 232, 40, 390, 132, 411, 375, 315, 334, 227, 4, 103, 303, 433, 122, 184, 36, 81, 5, 120, 28, 372, 73, 349, 49, 146, 43, 306, 147, 77, 400, 237, 316, 251, 243, 74, 212, 109, 171, 87, 52, 199, 21, 39, 420, 145, 293, 222, 183, 410, 418, 111, 173, 194, 16, 98, 93, 351, 260, 242, 317, 258, 426, 407, 170, 378, 181, 235, 248, 142, 177, 106, 185, 330, 336, 437, 436, 159, 277, 124, 69, 225, 33, 428, 353, 401, 282, 374, 204, 230, 197, 187, 51, 434, 100, 246, 311, 284, 318, 218, 245, 150, 423, 167, 203, 295, 140, 364, 377, 424, 297, 368, 201, 234, 304, 259, 37, 392, 128, 206, 333, 161, 215, 136, 55, 406, 432, 226, 131, 97, 382, 244, 46, 264, 42, 348, 309, 41, 80, 254, 129, 391, 178, 439, 59, 30, 275, 1, 283, 288, 229, 151, 205, 324, 313, 380, 34, 90, 186, 104, 408, 272, 76, 19, 105, 12, 322, 233, 143, 361, 99, 135, 56, 231, 379, 86, 270, 113, 45, 301, 267, 346, 61, 162, 127, 6, 190, 114, 3, 320, 339, 398, 393, 116, 431, 256, 345, 386, 405, 326, 412, 250, 362, 220, 153, 189, 280, 191, 427, 290, 117, 44, 219, 276, 137, 141, 174, 58, 358, 160, 94, 440, 118, 68, 271, 138, 396, 298, 67, 273, 223, 228, 60, 314, 278, 337, 384, 381, 399, 149, 325, 367, 47, 70, 22, 193, 281, 168, 126, 26, 359, 102, 50, 139, 148, 155, 196, 347, 294, 95, 209, 385, 403, 20, 365, 421, 417, 285, 332, 208, 101, 157, 64, 269, 419, 66, 344, 287, 125, 397, 266, 75, 179, 188, 413, 321, 370, 247, 214, 8, 310, 192, 29, 107, 13, 241, 402, 123, 286, 369, 291, 255, 166, 175, 82, 409, 268, 31, 323, 32, 9, 156, 414, 53, 224, 210, 85, 331, 79, 261, 63, 262, 133, 121, 252, 38, 211, 279, 180, 394, 35, 54, 165, 10, 65, 216, 292, 360, 200, 78, 221]


print("Imports done")
print("Directories:")
print("Root directory:", ROOT_DIR)
print("Model directory:", MODEL_DIR)
print("Pre-trained model directory:", PRETRAINED_MODEL_DIR)
print("M-RCNN directory:", MRCNN_DIR)
print("Datasets directory:", DATA_DIR)


class CoralConfigNewLayers(Config):
    """Configuration for training on the coral dataset.
    Derives from the base Config class and overrides values specific
    to the coral dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coral"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = BATCH_SIZE

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(CLASSES)

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    #RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    RPN_ANCHOR_SCALES = (48,  96, 192, 384, 768) if LARGER_IMAGES else (32, 64, 128, 256, 512) # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    LEARNING_RATE = LEARN_RATE_INITIAL

    LEARNING_MOMENTUM = 0.9

    WEIGHT_DECAY = 0.0001

    DETECTION_MIN_CONFIDENCE = 0.6

    MEAN_PIXEL = MEAN_PIXEL_CURRENT

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = STEPS_PER_EPOCH_TRAIN

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = STEPS_PER_EPOCH_VALIDATION

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (224, 224)


class CoralConfigAllLayers(Config):
    """Configuration for training on the coral dataset.
    Derives from the base Config class and overrides values specific
    to the coral dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coral"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = BATCH_SIZE

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(CLASSES)
    # NUM_CLASSES = 1 + 1

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

    LEARNING_RATE = LEARN_RATE_ALL_LAYERS

    LEARNING_MOMENTUM = 0.9

    WEIGHT_DECAY = 0.0001

    DETECTION_MIN_CONFIDENCE = 0.6

    MEAN_PIXEL = MEAN_PIXEL_CURRENT

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = STEPS_PER_EPOCH_TRAIN

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = STEPS_PER_EPOCH_VALIDATION

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (224, 224)

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


def map_polygon(points):
    if len(points) % 2 != 0:
        raise ValueError("Number of points must be even")

    for i in range(0, len(points), 2):
        points[i] = max(0, min(math.floor(int(points[i]) / 4032 * ORIGINAL_IMAGE_WIDTH), ORIGINAL_IMAGE_WIDTH) - 1)
        points[i+1] = max(0, min(math.floor(int(points[i+1]) / 3024 * ORIGINAL_IMAGE_HEIGHT), ORIGINAL_IMAGE_HEIGHT) - 1)

    return points


class CoralDataset(utils.Dataset):
    """Generates the bone age dataset.
    """
    def __init__(self, dataset_dir):
        super().__init__()
        self.dataset_dir = dataset_dir

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
                image_name=current_filename,
                annotations=annotations)

    def load_coral(self, annotation_file, image_indices):
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

            image_dir = self.dataset_dir
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
                m = np.full((image_info["height"], image_info["width"]), fill_value=False)
                # class, xmin, ymin, xmax, ymax -> ymin:ymax, xmin:xmax
                m[annotation[2]:annotation[4], annotation[1]:annotation[3]] = True

                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() == False:
                    #print("Skipped one!")
                    continue

                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        #masks = np.reshape(np.stack(instance_masks, axis=2).astype(np.bool), (
        #1, image_info["height"], image_info["width"], len(class_ids)))[0]
        masks = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)
        return masks, class_ids


class CoralSegmentationDataset(CoralDataset):
    """Generates the coral dataset.
    """

    def load_coral(self, annotation_file, image_indices):
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

            image_dir = self.dataset_dir
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


class Oversampler:
    def __init__(self, dataset, image_count_over_mean_multiplier = 2):
        self.dataset = dataset
        self.original_image_info = self.dataset.image_info.copy()
        self.image_count_over_mean_multiplier = image_count_over_mean_multiplier

    def get_class_counts(self):
        class_counts = {}
        for image_info in self.dataset.image_info:
            for a in image_info["annotations"]:
                class_name = a[0]

                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
        return class_counts

    def calculate_imbalance(self, class_counts):
        num_objects = np.sum(list(class_counts.values()))

        entropy = []
        for c in class_counts:
            entropy.append(class_counts[c] / num_objects * math.log10(class_counts[c] / num_objects))

        return np.sum(entropy) * -1

    def evaluate_dataset(self, print_classes = False):
        class_counts = self.get_class_counts()

        if print_classes:
            for c in class_counts:
                print(f"{c}: {class_counts[c]}")

        dataset_imbalance_value = self.calculate_imbalance(class_counts)

        if print_classes:
            print(f"Imbalance value: {dataset_imbalance_value}")

        # image values
        image_values = {}
        for image_info in self.original_image_info:
            class_counts_hypothetical = class_counts.copy()
            for a in image_info["annotations"]:
                class_counts_hypothetical[a[0]] += 1
            image_values[image_info["image_name"]] =\
                (image_info, self.calculate_imbalance(class_counts_hypothetical))

        sorted_image_values = {
            k: v for k, v in sorted(
                image_values.items(), key=lambda item: item[1][1], reverse=True)}
        #print(f"Highest value image: {sorted_image_values[list(sorted_image_values.keys())[0]][0]['image_name']} = {sorted_image_values[list(sorted_image_values.keys())[0]][1]}")

        return sorted_image_values

    def get_image_counts(self):
        image_counts = {}
        for image_info in self.dataset.image_info:
            if image_info["image_name"] not in image_counts:
                image_counts[image_info["image_name"]] = 0
            image_counts[image_info["image_name"]] += 1
        return image_counts

    def get_next_image_id(self):
        return self.dataset.image_info[-1]["id"] + 1

    def add_image(self):
        # get image evaluations
        evaluated_images = self.evaluate_dataset()
        # get image counts
        image_counts = self.get_image_counts()
        # get mean image count
        mean_image_count = np.mean(list(image_counts.values()))
        image_count_limit = math.ceil(self.image_count_over_mean_multiplier*mean_image_count)
        image_added = False
        # loop through sorted images
        for image_info, image_val in evaluated_images.values():
            # add first image with a valid count
            if image_counts[image_info["image_name"]] < image_count_limit:
                #print(f"Adding image {image_info['image_name']} = {image_val} ({image_counts[image_info['path']]})")
                self.dataset.add_coral_image([], self.get_next_image_id(),
                                             image_info["annotations"],
                                             self.dataset.dataset_dir,
                                             image_info["image_name"])
                image_added = True
                break
        return image_added

    def oversample(self, target_dataset_size):
        progress = progressbar.progressbar(range(target_dataset_size - len(self.dataset.image_info)))
        while len(self.dataset.image_info) < target_dataset_size and self.add_image():
            next(progress)

        if len(self.dataset.image_info) < target_dataset_size:
            raise RuntimeWarning("Oversampling failed")


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
    #elif init_with == "manual":
        #model.load_weights(MANUAL_MODEL_PATH, by_name=True)

    print("Model loaded")
    return model

def get_all_indices():
    return random_indices

def get_indices_train(i):
    return list(set(random_indices) - set(get_indices_val(i)))

def get_indices_val(i):
    return random_indices[i * 88: i * 88 + 88]


##
# End configuration and classes
##

with open(os.path.join(MODEL_DIR, "losses.csv"), 'a') as output_csv:
    writer = csv.writer(output_csv, delimiter=',')
    writer.writerow(['run_name', 'split', 'epoch', 'val_loss', 'train_loss', 'val_rpn_class_loss', 'val_rpn_bbox_loss', 'val_mrcnn_class_loss', 'val_mrcnn_bbox_loss', 'rpn_class_loss', 'rpn_bbox_loss', 'mrcnn_class_loss', 'mrcnn_bbox_loss'])

for split in range(5):

    print(f"Start split {split}")

    # Training dataset
    dataset_train = CoralSegmentationDataset(os.path.join(DATA_DIR, IMAGE_SUBDIR)) if SEGMENTATION else CoralDataset(os.path.join(DATA_DIR, IMAGE_SUBDIR))
    #dataset_train = CoralDataset(os.path.join(DATA_DIR, IMAGE_SUBDIR))
    dataset_train.load_coral(os.path.join(DATA_DIR, ANNOTATION_CSV),
                            get_indices_train(split) if CROSS_VALIDATION else get_all_indices())
    if OVERSAMPLING:
        oversampler = Oversampler(dataset_train, 2)
        oversampler.evaluate_dataset(print_classes=True)
        oversampler.oversample(3 * len(dataset_train.image_info))
        oversampler.evaluate_dataset(print_classes=True)

        sorted_image_counts = {
            k: v for k, v in sorted(
                oversampler.get_image_counts().items(),
                key=lambda item: item[1], reverse=True)}
        j = 0
        for image_name in sorted_image_counts:
            print(f"#{j + 1} {image_name}: {sorted_image_counts[image_name]}")
            j += 1
            if sorted_image_counts[image_name] == 1:
                break

    dataset_train.prepare()
    print("Images in training set: " + str(len(dataset_train.image_info)))

    dataset_val = CoralSegmentationDataset(os.path.join(DATA_DIR, IMAGE_SUBDIR)) if SEGMENTATION else CoralDataset(os.path.join(DATA_DIR, IMAGE_SUBDIR))
    #dataset_val = CoralDataset(os.path.join(DATA_DIR, IMAGE_SUBDIR))
    dataset_val.load_coral(os.path.join(DATA_DIR, ANNOTATION_CSV),
                            get_indices_val(split) if CROSS_VALIDATION else get_all_indices())
    dataset_val.prepare()
    print("Images in validation set: " + str(len(dataset_val.image_info)))

    configNewLayers = CoralConfigNewLayers()
    print('Config for training the new layers:')
    configNewLayers.display()

    history = LossHistory()

    model = create_model("training", configNewLayers, "coco", split)

    # Train the head branches
    model.train(dataset_train, dataset_val,
                learning_rate=configNewLayers.LEARNING_RATE,
                epochs=1,
                custom_callbacks=[history],
                layers='heads')

    print(f"Training of new layers for split {split} finished")

    configAllLayers = CoralConfigAllLayers()
    print('Config for training all layers:')
    configAllLayers.display()

    model = create_model("training", configAllLayers, "last", split)

    augmentation = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.33), # horizontally/vertically flip 50% of all images
            iaa.Flipud(0.33),
            iaa.Sometimes(0.33, iaa.Affine(rotate=(-180, 180))),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.CropAndPad(percent=(-0.25, 0)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                ],
                random_order=True
            )
        ],
        random_order=True
    )

    #augmentation = iaa.Sometimes(0.75, iaa.SomeOf((1, None), [
    #   iaa.Fliplr(0.5),
    #   iaa.Flipud(0.5),
    #   iaa.Affine(rotate=(-180, 180)),
    #   iaa.CropAndPad(percent=(-0.25, 0))
    #]))

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=configAllLayers.LEARNING_RATE,
                epochs=EPOCHS,
                custom_callbacks=[history],
                augmentation=augmentation if AUGMENTATION else None,
                layers="all")

    print(f"Training of all layers for split {split} finished")

    tf.keras.backend.clear_session()

    if not CROSS_VALIDATION:
        break
