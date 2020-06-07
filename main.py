"""
Name: Background remove tool.
Description: This file contains the CLI interface.
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
import argparse
import os
import tqdm
import logging
from libs.strings import *
from libs.networks import model_detect

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def __work_mode__(path: str):
    """Determines the desired mode of operation"""
    if os.path.isfile(path):  # Input is file
        return "file"
    if os.path.isdir(path):  # Input is dir
        return "dir"
    else:
        return "no"


def __save_image_file__(img, file_name, output_path, wmode):
    """
    Saves the PIL image to a file
    :param img: PIL image
    :param file_name: File name
    :param output_path: Output path
    :param wmode: Work mode
    """
    if wmode == "file":
        file_name_out = os.path.basename(output_path)
        if file_name_out == '':
            # Change file extension to png
            file_name = os.path.splitext(file_name)[0] + '.png'
            # Save image
            img.save(os.path.join(output_path, file_name))
        else:
            try:
                # Save image
                img.save(output_path)
            except OSError as e:
                raise OSError("Error! "
                              "Please indicate the correct extension of the final file, for example: .png",
                              "Error: ", e)
    else:
        # Change file extension to png
        file_name = os.path.splitext(file_name)[0] + '.png'
        # Save image
        img.save(os.path.join(output_path, file_name))


def cli():
    """CLI"""
    parser = argparse.ArgumentParser(description=DESCRIPTION, usage=ARGS_HELP)
    parser.add_argument('-i', required=True,
                        help="Path to input file or dir.", action="store", dest="input_path")
    parser.add_argument('-o', required=True,
                        help="Path to output file or dir.", action="store", dest="output_path")
    parser.add_argument('-m', required=False,
                        help="Model name. Can be {} . U2NET is better to use.".format(MODELS_NAMES),
                        action="store", dest="model_name", default="u2net")
    args = parser.parse_args()
    # Parse arguments
    input_path = args.input_path
    output_path = args.output_path
    if input_path is None or output_path is None:
        raise Exception("Bad parameters! Please specify input path and output path.")
    model_name = args.model_name
    model = model_detect(model_name)  # Load model
    if not model:
        logger.warning("Warning! You specified an invalid model type. "
                       "For image processing, the model with the best processing quality will be used. "
                       "(u2net)")
        model_name = "u2net"  # If the model line is wrong, select the model with better quality.
        model = model_detect(model_name)  # Load model

    wmode = __work_mode__(input_path)  # Get work mode
    if wmode == "file":  # File work mode
        image = model.process_image(input_path)
        __save_image_file__(image, os.path.basename(input_path), output_path, wmode)
    elif wmode == "dir":  # Dir work mode
        # Start process
        files = os.listdir(input_path)
        for file in tqdm.tqdm(files, ascii=True, desc='Remove Background', unit='image'):
            file_path = os.path.join(input_path, file)
            image = model.process_image(file_path)
            __save_image_file__(image, file, output_path, wmode)
    else:
        raise Exception("Bad input parameter! Please indicate the correct path to the file or folder.")


if __name__ == "__main__":
    cli()
