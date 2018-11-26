"""
Mask R-CNN for Cetacean Photogrammatry

Written by Patrick Gray


Based on Balloon Color Splash Project by Waleed Abdulla
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 .py train --dataset=/path/to/whale/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 whale.py train --dataset=/path/to/whale/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 whale.py train --dataset=/path/to/whale/dataset --weights=imagenet

    # Apply color splash to an image
    python3 whale.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 whale.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa

# Root directory of the project
MASK_RCNN_DIR = os.path.abspath("../Mask_RCNN/")
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(MASK_RCNN_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(MASK_RCNN_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class WhaleConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "whale"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2 + 2 + 2 # Background + blue body/pec + humpback body/pec + minke body/pec

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 265 # num of total training images
    #STEPS_PER_EPOCH = 265*6 # num of total training images

    # Skip detections with < 60% confidence
    DETECTION_MIN_CONFIDENCE = 0.6

     # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    # default
    #MASK_SHAPE = [28, 28]
    # increasing resolution
    MASK_SHAPE = [56, 56]
    #MASK_SHAPE = [112, 112]

    # Length of square anchor side in pixels
    # Making larger since objects and images are large
    RPN_ANCHOR_SCALES = (128, 256, 512)

    #RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 2

    #TRAIN_ROIS_PER_IMAGE = 200
    #TRAIN_ROIS_PER_IMAGE = 100
    # reduce ROIs because we have few objects per iamge
    TRAIN_ROIS_PER_IMAGE = 32

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.001


    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 2.,
        "mrcnn_class_loss": 2.,
        "mrcnn_bbox_loss": 2.,
        "mrcnn_mask_loss": 5.
    }


############################################################
#  Dataset
############################################################

class WhaleDataset(utils.Dataset):

    def load_whale(self, dataset_dir, subset):
        """Load a subset of the Whale dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("whale", 1, "balaenoptera_musculus_body")
        self.add_class("whale", 2, "balaenoptera_musculus_pectoral")
        self.add_class("whale", 3, "megaptera_novaeangliae_body")
        self.add_class("whale", 4, "megaptera_novaeangliae_pectoral")
        self.add_class("whale", 5, "balaenoptera_acutorostrata_body")
        self.add_class("whale", 6, "balaenoptera_acutorostrata_pectoral")

        # Train, validation, or test dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygon_list = list(a['regions'].values())

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            try:
                height, width = image.shape[:2]
            except ValueError:
                height, width = image[0].shape[:2]
            self.add_image(
                "whale",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygon_list=polygon_list)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a whale dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != "whale":
            return super(self.__class__, self).load_mask(image_id)

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        mask = np.zeros([info["height"], info["width"], len(info["polygon_list"])],
                        dtype=np.uint8)

        class_ids = []
        for i, p in enumerate(info["polygon_list"]):
            class_id = 0
            
            # determine the class based on the region attributes in the json dict
            if p['region_attributes']['species'] == "balaenoptera_musculus":
                class_id = 0
            elif p['region_attributes']['species'] == "megaptera_novaeangliae":
                class_id = 2
            elif p['region_attributes']['species'] == "balaenoptera_acutorostrata":
                class_id = 4

            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['shape_attributes']['all_points_y'], p['shape_attributes']['all_points_x'])
            if p['region_attributes']['body_part'] == "body":
                try:    # some of the masks were right on the edge of the image and thus I needed to catch those cases
                    mask[rr, cc, i] = 1     # this sets the mask = True for this area
                except IndexError:
                    mask[rr-1, cc-1, i] = 1
                class_id = class_id+1   # this adds to the previously determined class to get the actual class_id
            elif p['region_attributes']['body_part'] == "pectoral":
                try:
                    mask[rr, cc, i] = 1     # this sets the mask = True for this area
                except IndexError:
                    mask[rr-1, cc-1, i] = 1
                class_id = class_id+2   # this adds to the previously determined class to get the actual class_id

            class_ids.append(class_id)

        # Return mask, and array of class IDs of each instance. 
        return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])

        # if image has issues with not having [H,W,C] fix it
        try:
            height, width, channels = image.shape
        except ValueError:
            image = image[0]

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "whale":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = WhaleDataset()
    dataset_train.load_whale(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = WhaleDataset()
    dataset_val.load_whale(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.

    # first run
    #layers_training='heads'
    # second run
    #layers_training='3+'
    # third run
    layers_training='all'
    epochs_to_train=250

    # adding image augmentation parameters

    augmentation = iaa.Sometimes(.667, iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.25.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.25))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2)),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-180, 180),
            #shear=(-8, 8)
        )
    ], random_order=True)) # apply augmenters in random order

    # old image aug parameters
    """
    augmentation = iaa.Sometimes(0.9, [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Multiply((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
        ])
    """

    print("Training heads with augmentation.")
    print("*****Beginning training*****")
    print("config.LEARNING_RATE", config.LEARNING_RATE/5)
    print("layers_training:", layers_training)
    print("epochs_to_train:", epochs_to_train)
    print("augmentation: ", augmentation)
    print("---")
    print("Images: {}\nClasses: {}".format(len(dataset_train.image_ids), dataset_train.class_names))
    #print("Training network in its entirety with augmentation")
    
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/5,
                epochs=epochs_to_train,
                layers=layers_training,
                augmentation=augmentation)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect whales.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/whale/dataset/",
                        help='Directory of the Whale dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = WhaleConfig()
    else:
        class InferenceConfig(WhaleConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
