"""
Name: Strings file
Description: This file contains the strings.
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

from libs.networks import MODELS_NAMES
NAME = "Image Background Remove Tool"
DESCRIPTION = "A tool to remove a background from image using Neural Networks"
LICENSE = "Apache License 2.0"
ARGS_HELP = """
{}
{}
License: {}
Running the script:
python3 main.py -i <input_path> -o <output_path> -m <model_type>
Explanation of args:
-i <input_path> - path to input file or dir.
-o <output_path> - path to output file or dir.
-m <model_type> - can be {}. U2NET is better to use. 
DeepLab models (xception_model or mobile_net_model) are outdated 
and designed to remove the background from PORTRAIT photos or PHOTOS WITH ANIMALS! 
""".format(NAME, DESCRIPTION, LICENSE, ' or '.join(MODELS_NAMES))