"""
Name: Neural networks file.
Description: This file contains neural network classes.
Version: [release][3.1]
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Anodev (OPHoperHPO)[https://github.com/OPHoperHPO] .
License: Apache License 2.0
License:
   Copyright 2020 OPHoperHPO

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import logging
import os
import time
from skimage import io, transform
import numpy as np
from PIL import Image
from io import BytesIO

tf = None  # Define temp modules
ndi = None
torch = None
Variable = None
U2NET_DEEP = None
U2NETP_DEEP = None

logger = logging.getLogger(__name__)
MODELS_NAMES = ["u2net", "u2netp", "xception_model", "mobile_net_model"]


# noinspection PyUnresolvedReferences
def model_detect(model_name):
    """Detects which model to use and returns its object"""
    global tf, ndi, torch, Variable, U2NET_DEEP, U2NETP_DEEP
    if model_name in MODELS_NAMES:
        if model_name == "xception_model" or model_name == "mobile_net_model":
            import scipy.ndimage as ndi
            import tensorflow as tf
            return TFSegmentation(model_name)
        elif "u2net" in model_name:
            import torch
            from torch.autograd import Variable
            from libs.u2net import U2NET as U2NET_DEEP
            from libs.u2net import U2NETP as U2NETP_DEEP
            return U2NET(model_name)
    else:
        return False


# noinspection PyUnresolvedReferences
class U2NET:
    """U^2-Net model interface"""

    def __init__(self, name="u2net"):
        if name == 'u2net':  # Load model
            logger.debug("Loading a U2NET model (176.6 mb) with better quality but slower processing.")
            net = U2NET_DEEP()
        elif name == 'u2netp':
            logger.debug("Loading a U2NETp model (4 mb) with lower quality but fast processing.")
            net = U2NETP_DEEP()
        else:
            logger.debug("Loading a Unknown model!")
        try:
            if torch.cuda.is_available():
                net.load_state_dict(torch.load(os.path.join("models", name, name + '.pth')))
                net.cuda()
            else:
                net.load_state_dict(torch.load(os.path.join("models", name, name + '.pth'), map_location="cpu"))
        except FileNotFoundError:
            raise FileNotFoundError("No pre-trained model found! Run setup.sh or setup.bat to download it!")
        net.eval()
        self.__net__ = net  # Define model

    def process_image(self, path):
        """
        Removes background from image and returns PIL RGBA Image.
        :param path: Path to image
        :return: PIL RGBA Image. If an error reading the image is detected, returns False.
        """
        start_time = time.time()  # Time counter
        logger.debug("Load image: {}".format(path))
        image, org_image = self.__load_image__(path)  # Load image
        if image is False or org_image is False:
            return False
        image = image.type(torch.FloatTensor)
        if torch.cuda.is_available():
            image = Variable(image.cuda())
        else:
            image = Variable(image)
        mask, d2, d3, d4, d5, d6, d7 = self.__net__(image)  # Predict mask
        logger.debug("Mask prediction completed")
        # Normalization
        logger.debug("Mask normalization")
        mask = mask[:, 0, :, :]
        mask = self.__normalize__(mask)
        # Prepare mask
        logger.debug("Prepare mask")
        mask = self.__prepare_mask__(mask, org_image.size)
        # Apply mask to image
        logger.debug("Apply mask to image")
        empty = Image.new("RGBA", org_image.size)
        image = Image.composite(org_image, empty, mask)
        logger.debug("Finished! Time spent: {}".format(time.time() - start_time))
        return image

    def __load_image__(self, path: str):
        """
        Loads an image file for other processing
        :param path: Path to image file
        :return: image tensor, original image shape
        """
        image_size = 320  # Size of the input and output image for the model
        try:
            image = io.imread(path)  # Load image
        except IOError:
            logger.error('Cannot retrieve image. Please check file: ' + path)
            return False, False
        pil_image = Image.fromarray(image)
        image = transform.resize(image, (image_size, image_size), mode='constant')  # Resize image
        image = self.__ndrarray2tensor__(image)  # Convert image from numpy arr to tensor
        return image, pil_image

    @staticmethod
    def __ndrarray2tensor__(image: np.ndarray):
        """
        Converts a NumPy array to a tensor
        :param image: Image numpy array
        :return: Image tensor
        """
        tmp_img = np.zeros((image.shape[0], image.shape[1], 3))
        image /= np.max(image)
        if image.shape[2] == 1:
            tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmp_img[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        tmp_img = tmp_img.transpose((2, 0, 1))
        tmp_img = np.expand_dims(tmp_img, 0)
        return torch.from_numpy(tmp_img)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def __normalize__(predicted):
        """Normalize the predicted map"""
        ma = torch.max(predicted)
        mi = torch.min(predicted)
        out = (predicted - mi) / (ma - mi)
        return out

    @staticmethod
    def __prepare_mask__(predict, image_size):
        """Prepares mask"""
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()
        mask = Image.fromarray(predict_np * 255).convert("L")
        mask = mask.resize(image_size, resample=Image.BILINEAR)
        return mask


class TFSegmentation(object):
    """Class to load deeplab model and run inference."""

    # noinspection PyUnresolvedReferences
    def __init__(self, model_type):
        """Creates and loads pretrained deeplab model."""
        # Environment init
        self.INPUT_TENSOR_NAME = 'ImageTensor:0'
        self.OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
        self.INPUT_SIZE = 513
        self.FROZEN_GRAPH_NAME = 'frozen_inference_graph'
        # Start load process
        self.graph = tf.Graph()
        try:
            graph_def = tf.compat.v1.GraphDef.FromString(open(os.path.join("models", model_type, "model",
                                                                           "frozen_inference_graph.pb"),
                                                              "rb").read())
        except FileNotFoundError:
            raise FileNotFoundError("No pre-trained model found! Run setup.sh or setup.bat to download it!")
        logger.warning("Loading a DeepLab model ({})! "
                       "This is an outdated model with poorer image quality and processing time."
                       "Better use the U2NET model instead of this one!".format(model_type))
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.compat.v1.Session(graph=self.graph)

    def process_image(self, path: str):
        """
        Removes background from image and returns PIL RGBA Image.
        :param path: Path to image
        :return: PIL RGBA Image. If an error reading the image is detected, returns False.
        """
        start_time = time.time()  # Time counter
        logger.debug('Load image')
        try:
            jpeg_str = open(path, "rb").read()
            image = Image.open(BytesIO(jpeg_str))
        except IOError:
            logger.error('Cannot retrieve image. Please check file: ' + path)
            return False
        seg_map = self.__predict__(image)
        logger.debug('Finished mask creation')
        image = image.convert('RGB')
        logger.debug("Mask overlay completed")
        image = self.__draw_segment__(image, seg_map)
        logger.debug("Finished! Time spent: {}".format(time.time() - start_time))
        return image

    def __predict__(self, image):
        """Image processing."""
        # Get image size
        width, height = image.size
        # Calculate scale value
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        # Calculate future image size
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        # Resize image
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        # Send image to model
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        # Get model output
        seg_map = batch_seg_map[0]
        # Get new image size and original image size
        width, height = resized_image.size
        width2, height2 = image.size
        # Calculate scale
        scale_w = width2 / width
        scale_h = height2 / height
        # Zoom numpy array for original image
        seg_map = ndi.zoom(seg_map, (scale_h, scale_w))
        return seg_map

    @staticmethod
    def __draw_segment__(image, alpha_channel):
        """Postprocessing. Saves complete image."""
        # Get image size
        width, height = image.size
        # Create empty numpy array
        dummy_img = np.zeros([height, width, 4], dtype=np.uint8)
        # Create alpha layer from model output
        for x in range(width):
            for y in range(height):
                color = alpha_channel[y, x]
                (r, g, b) = image.getpixel((x, y))
                if color == 0:
                    dummy_img[y, x, 3] = 0
                else:
                    dummy_img[y, x] = [r, g, b, 255]
        # Restore image object from numpy array
        img = Image.fromarray(dummy_img)
        return img
