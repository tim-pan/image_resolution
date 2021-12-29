# image super resolution
## task
image super resolution with upscaling factor=3</br>
training dataset:291 HR images</br>
testing dataset:14 LR images</br>
## Environment
- colab GPU(Tesla P100)
## packages
- pytorch
- numpy
- Pillow
- torchvision
- scikit-learn
- scikit-image
- opencv-python
- tensorboard
## introduction
there are 
- 10 files</br>
**1. datasets.py**</br>
define the custom dataset </br>
**2. eval.py**</br>
evaluate the validation set</br>
**3. main.ipynb**</br>
run training procedure</br>
**4. model.py**</br>
SRResnet architecture in this file</br>
**5. swinIR.py**</br>
implementation of swinLR, is fromthe official totorial of [4]</br>
**6. test.py**</br>
use model to do the image super resolution</br>
**7. train.py**</br>
implement training procedure,I also modified its code to fit our experiments</br>
**8. train_val_split.py**</br>
split dataset to training dataset and validation dataset</br>
**9. utils.py**</br>
something we need but not included in any py files above</br>
**10. inference.ipynb**</br>
reproduce the submission</br>
- 2 folders</br>
**1. data**</br>
the place where your testing data and training data is</br>
**2. models**</br>
This folder used to put the trained model weights</br> 
## reproduce
- you only need to download `inference.ipynb` and just run it on colab.

**note** </br>
The model para you need to reproduce the submission are embedded in the `inference.ipynb`, but if you want to check my weight para is normal, please follow this [link](https://drive.google.com/file/d/1cdSDdbzfGp97Ijvol2CKBSsTF_w0PpBz/view?usp=sharing)
## reference
[1] https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution</br>
[2] Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.By C.Ledig et al.</br> 
[3] SwinIR: Image Restoration Using Swin Transformer.By J.Liang et al.</br>
[4] https://github.com/jingyunliang/swinir </br>
[5] A comprehensive review of deep learning based single image super-resolution. By Syed Muhammad Arsalan Bashir et al.</br>
[6] Image Super-Resolution Using Deep Convolutional Networks. By Chao Dong et al.</br>
[7] Deep learning based single image super-resolution: a survey. International Journal of Automation and Computing, by Ha et al.</br>
[8] Enhanced Deep Residual Networks for Single Image Super-Resolution.By Bee Lim, et al.</br>




