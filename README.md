# 🥧 Image Background Remove Tool 🥧
Tool for removing background from image using neural networks
**********************************************************************
### 📄 Description:
The program removes the background from photos
**********************************************************************
### 🎆 Features:
* **Added support for new neural networks ([U^2-NET](https://github.com/NathanUA/U-2-Net)) on PyTorch**
* **Significantly improved output image quality**
* __Tensorflow 2.0 compatible__
* __All models support processing both on the video card and on the processor.__
* ```tqdm``` progress bar.
* __Removes background from image without loss of image resolution.__
*  __The script not only processes a single file, but can also process all images from the input folder and save them in the output folder with the same name.__
*  __Implemented support for the neural network from this [ script](https://github.com/susheelsk/image-background-removal) and improved the result of its work__

**********************************************************************
### 🧷 Dependencies:
```	gdown ``` **for setup.py!** \
```	tensorflow, torch, Pillow, tqdm, numpy, scipy, scikit_image ``` **for main.py!**
> Note:  You can choose what to install PyTorch or TensorFlow, based on which model you want to use. \
PyTorch for `u2net`, `u2netp` \
TensorFlow for `xception_model`, `mobile_net_model` \
**TensorFlow models are not recommended for use, since these models have much worse quality and lower image processing speed, also these models are designed solely to remove the background from portrait photos and photos with animals.**
**********************************************************************
### 🏷 Setup for Windows:
* Clone this repository
* Install all the dependencies from **requirements.txt** via ```pip3 install -r requirements.txt```
* Run ```./setup.bat``` \
_This setup.bat script loads the trained model._
### 🏷 Setup for Linux:
* Clone repository: ```git clone https://github.com/OPHoperHPO/image-background-remove-tool```
* Install all the dependencies from **requirements.txt**: ```pip3 install -r requirements.txt```
* Run ```./setup.sh``` and select the model you need.\
_This setup.sh script loads the trained model._
**********************************************************************
### 🧰 Running the script:
 * ```python3 main.py -i <input_path> -o <output_path> -m <model_type>```
#### Explanation of args:
 * `-i <input_path>` - path to input file or dir.
 * `-o <output_path>` - path to output file or dir.
 * `-m <model_type>` - can be `u2net` or `u2netp` or `xception_model` or `mobile_net_model`. `u2net` is better to use. 
__DeepLab__ models (`xception_model` or `mobile_net_model`) are __outdated__ 
and designed to remove the background from PORTRAIT photos or PHOTOS WITH ANIMALS! \
[More info about models.](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/docs/MODELS.md)
 > Note:  See example scripts for more information on using the program.
**********************************************************************
### ⏳ TODO:
```
1) Add a graphical interface. (0% done)
```
### 💵 Support me:

You can thank me for developing this project, provide financial support for the development of new projects and buy me a small cup of coffee.☕\
  Just support me on these platforms: \
  ⭐[**Boosty**⭐](https://boosty.to/anodev) \
  ⭐[**DonationAlerts**⭐](https://www.donationalerts.com/r/anodev_development)
### 😀 Sample Result:
* __More sample images in [docs/imgs/input/](https://github.com/OPHoperHPO/image-background-remove-tool/tree/master/docs/imgs/input) and [docs/imgs/examples/](https://github.com/OPHoperHPO/image-background-remove-tool/tree/master/docs/imgs/examples) folders__
* Input: 
* ![Input](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/docs/imgs/input/1.jpg "Input")

* Output(u2net): 
* ![Output](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/docs/imgs/examples/u2net/1.png "Output")
* Output(u2netp): 
* ![Output](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/docs/imgs/examples/u2netp/1.png "Output")
* Output(xception_model): 
* ![Output](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/docs/imgs/examples/xception_model/1.png "Output")
* Output(mobile_net_model): 
* ![Output](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/docs/imgs/examples/mobile_net_model/1.png "Output")
**********************************************************************
