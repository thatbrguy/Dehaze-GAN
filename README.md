# Dehaze-GAN
This repository contains TensorFlow code for the paper titled A Single Image Haze Removal using a Generative Adversarial Network. [Demo](Youtube!)

(image)

**Note:** 
1. The first version of this project was completed around December 2017. The demo video (dated March 2018) reflects the performance of one of the final versions, however some iterative improvements were made after that. 
2. This repository contains code that can be used for any application, and is not limited to Dehazing. 
3. For recreating the results reported in the paper, use the repository `legacy` (for more details refer below). This repository is the refactored version of the final model, but it uses newer versions of some TensorFlow operations. Those operations are not available in the old saved checkpoints.

## Features:
The model has the following components:
- The 56-Layer Tiramisu as the generator.
- A patch-wise discriminator.
- A weighted loss function involving three components, namely:
  - GAN loss compoenent.
  - Perceptual loss component (aka VGG loss component).
  - L1 loss component.

The GAN loss component is dervied from the pix2pix GAN paper. Perceptual loss involves using only the (what) component of (this work). 

## Requirements:
- TensorFlow (version 1.4+)
- Matplotlib
- Numpy
- Scikit-Image

## Instructions:
1. Clone the repository using:
```
git clone https://github.com/thatbrguy/Dehaze-GAN.git
```
2. A VGG-19 pretrained on the ImageNet dataset is required to calculate Perceptual loss. In this work, we used the weights provided by [link](placeholder)'s implementation. Download the weights from this [link](placeholder) and include it in this repository.
> **Note:** You can use Keras' pretrained VGG-19 as well, as it can automatically download the ImageNet weights. However, my original implementation did not use it.

3. Train the model by using:
```
python train.py
```
The file `train.py` supports a lot of options, which are listed below:
- `--model_name`: Tensorboard, logs, samples and checkpoint files are stored in a folder named `model_name`. Default value is `model`.
- `--lr`: Sets the learning rate for both the generator and the discriminator. Default value is `0.001`.
- `--epochs`: Sets the number of epochs. Default value is `200`.
- `--batch_size`: Sets the batch_size. Default value is `1`.
- `--restore`: Boolean flag that enables restoring from old checkpoint. Checkpoints must be stored at `model_name/checkpoint`. Default value is `False`.
- `--gan_wt`: Weight factor for the GAN loss. Default value is `2`.
- `--l1_wt`: Weight factor for the L1 loss. Default value is `100`.
- `--vgg_wt`: Weight factor for the Perceptual loss (VGG loss). Default value is `10`.
- `--growth_rate`: Growth rate of the dense block. Refer to the DenseNet paper to learn more. Default value is `12`.
- `--layers`: Number of layers per dense block. Default value is `4`.
- `--decay`: Decay for the batchnorm operation. Default value is `0.99`.
- `--D_filters`: Number of filters in the 1st conv layer of the discriminator. Number of filters is multiplied by 2 for every successive layer. Default value is `64`.


## License:
This repository is open source under the MIT clause. Feel free to use it for academic and educational purposes.
