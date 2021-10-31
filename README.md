# Dehaze-GAN
This repository contains TensorFlow code for the paper titled **Single Image Haze Removal using a Generative Adversarial Network.** [[Demo](https://www.youtube.com/watch?v=ioSL6ese46A)][[Arxiv](http://arxiv.org/abs/1810.09479)]

<p align = "center">
	<img src="/src/fog.png" alt="Dehaze-GAN in action" height=50% width=50%>
	</img>
</p>

## Update (June 2021) (important)
On a recent review of some code I found a few issues. In the document [issue.md](issue.md), I have provided an explanation of the issues, their possible impact, and the remedies I have taken to account for the issues. Please read the document before attempting to use the codebase. I apologize for the inconvenience.

## Update (August 2020)
This work has been accepted for the 2020 International Conference on Wireless Communications, Signal Processing and Networking (**WiSPNET 2020**). The arxiv version of this work (originally published around 2018) will be updated with the accepted version of the paper. ~~Will attach a document listing out the major updates to the paper soon!~~

> (EDIT: Oct 2021) I wanted to upload a document describing the major changes between v1 and v2 of the arxiv papers but I was not able to allocate enough time to do so. The v1 paper was put on arxiv sometime in October 2018. The v2 version is shorter, but is better written than the v1 version and also has made some corrections. The v2 version of the paper is the one that was accepted for the WiSPNET 2020 conference. The interested reader can access both versions from arxiv. However, it is recommended to follow the v2 version of the paper (along with `issue.md` and other docs in this codebase).

## Features:
The model has the following components:
- The 56-Layer Tiramisu as the generator.
- A patch-wise discriminator.
- A weighted loss function involving three components, namely:
  - GAN loss component.
  - Perceptual loss component (aka VGG loss component).
  - L1 loss component.

Please refer to the [paper](http://arxiv.org/abs/1810.09479) for a detailed description. 

<p align = "center">
	<img src="/src/model.png" alt="Block diagram of the Dehaze-GAN">
	</img>
</p>

## Notes: 
1. The first version of this project was completed around December 2017. The demo video (dated March 2018) reflects the performance of one of the final versions, however some iterative improvements were made after that. 
2. This repository contains code that can be used for any application, and is not limited to Dehazing. 
3. For recreating the results reported in the paper, use the repository `legacy` (for more details refer below). This repository is the refactored version of the final model, but it uses newer versions of some TensorFlow operations. Those operations are not available in the old saved checkpoints.
4. ~~The codebase uses OpenCV's `imread` and `imwrite` functions without converting them from BGR to RGB space. However, there might be cases where this type of usage (such as was raised in this issue about `extract.py`) may not be desirable. To maintain reproducibility, the original code is left intact. If you application desires usage of images in the RGB space, you could manually convert them from BGR to RGB.~~ (**UPDATE: June 2021**) Please ignore the content that is stricken through. On a recent analysis I made a few key observations. Please refer to the document [issue.md](issue.md) for a detailed explanation of the colorspace issue.

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

2. The codebase has two branches namely `master` and `alt`. There are some differences between the two branches. The user has to choose one of the two branches to use based on their preferences. Please read [issue.md](issue.md) to understand the differences between the two branches. For instructions on the steps that are needed to be taken to setup the desired branch, go the section `Step 2 from README Instructions` in `issue.md` ([link](issue.md/#step-2-from-readme-instructions)).

3. A VGG-19 pretrained on the ImageNet dataset is required to calculate perceptual loss. In this work, we used the weights provided by [machrisaa](https://github.com/machrisaa/tensorflow-vgg)'s implementation. Download the weights from this [link](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) and include it in this repository.
> **Note:** You may consider using a different implementation of imagenet pretrained VGG-19 which can automatically download the weights (for example, the implementation from keras). However, in that case you would also have to modify the codebase a lot so that it can correctly use your desired implementation. Hence, please be careful if you want to use a different implementation.

4. Download the dataset.
- We used the [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and the [Make 3D](http://make3d.cs.cornell.edu/data.html) dataset for training. The following code will download the NYU Depth Dataset V2 and create the hazy and clear image pairs. The images will be placed in directories `A` and `B` respectively. The formula and values used to create the synthetic haze are stated in our paper.
```
wget -O data.mat http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
python extract.py
```
> **Note 1:** The above step is only given for the NYU dataset. If you are interested in creating hazy and clear image pairs for any other dataset which has RGB and depth information, please create your own script. The method mentioned in `extract.py` can be adapted for your custom dataset.

> **Note 2:** For training the model in the paper (and also for validation and some testing experiments), we also used some data from the Make 3D dataset. Around the time the codebase was released, I think there were some issues with accessing the dataset link and hence `extract.py` was only given for the NYU dataset. If the link is accessible now and you are interested in using the Make 3D dataset, you can create your own script to create hazy and clear image pairs by using the method in `extract.py` as a reference.

5. In case you want to use your own dataset, follow these instructions. If not, skip this step.
- Create two directories `A` and `B` in this repository. 
- Place the input images into directory `A` and target images into directory `B`. 
- Ensure that an input and target image pair has the **same name**, otherwise the program will throw an error (For instance, if `0001.png` is present in `A` it must also be present in `B`). 
- Resize all images to be of size `(256, 256, 3)`.

6. Train the model by using the following code. 
```
python main.py \
--A_dir A \
--B_dir B \
--batch_size 2 \
--epochs 20
```
The file `main.py` supports a lot of options, which are listed below:
- `--mode`: Select between `train`, `test` and `inference` modes. For `test` and `inference` modes, please place the checkpoint files at `./model/checkpoint` (you can replace `model` with your setting of the `--model_name` argument). Default value is `train`.
- `--model_name`: Tensorboard, logs, samples and checkpoint files are stored in a folder named `model_name`. This argument allows you to provide the name of that folder. Default value is `model`.
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
- `--save_samples`: Since GAN convergence is hard to interpret from metrics, you can choose to visualize the output of the generator after each validation run. This boolean flag enables the behavior. Default value is `False`.
- `--sample_image_dir`: If `save_samples` is set to `True`, you must provide sample images placed in a directory. Give the name of that directory to this argument. Default value is `samples`.
- `--custom_data`: Boolean flag that allows you to use your own data for training. Default is `True`. (Note: As of now, I have not linked the data I used for training).
- `--A_dir`: Directory containing the ipnut images. Only used when `custom_data` is set to `True`. Default value is `A`.
- `--B_dir`: Directory containing the target images. Only used when `custom_data` is set to `True`. Default value is `B`.
- `--val_fraction`: Fraction of the data to be used for validation. Only used when `custom_data` is set to `True`. Default value is `0.15`.
- `--val_threshold`: Number of steps to wait before validation is enabled. Usually, the GAN performs suboptimally for quite a while. Hence, disabling validation initially can prevent unnecessary validation and speeds up training. Default value is `0`.
- `--val_frequency`: Number of batches to wait before performing the next validation run. Setting this to `1` will perform validation after one discriminator and generator step. You can set it to a higher value to speed up training. Default value is `20`.
- `--logger_frequency`: Number of batches to wait before logging the next set of loss values. Setting it to a higher value will reduce clutter and slightly increase training speed. Default value is `20`.

## Replicating:
The code in `legacy` can be used for replicating results of our model on the test split of the "Custom Dataset" as mentioned in the paper. The following steps explain how to replicate the results:

1. Download the model checkpoint and the data used for testing from this [link](https://drive.google.com/file/d/1d2HUyumIu6BwSYiPuGOdnvTiCArQjrCm/view?usp=sharing). Place the tar file inside the `legacy` folder. Extract the contents using the following commands.
```
cd legacy
tar -xzvf replicate.tar.gz
``` 
2. Move weights of the pretrained VGG-19 into the the `legacy` folder. The download [link](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) is reproduced here for convenience.

3. Run the code from the `legacy` folder using:
```
python replicate.py
```

If you would like to replicate the results on the SOTS outdoor dataset as well, you may do so. However, a ready to use script to quickly test it is not provided. However, it should be pretty straightforward to write and add a small function in `replicate.py` to test the same.

## Known Issues:
1. For newer versions of NumPy, you may get an error when the codebase attempts to load the pretrained VGG-19 weights NumPy file (for example, see issue [#7](https://github.com/thatbrguy/Dehaze-GAN/issues/7)). One simple solution to avoid this issue is to use a lower version of NumPy (as mentioned in issue [#7](https://github.com/thatbrguy/Dehaze-GAN/issues/7)). If you prefer to use a higher version of NumPy, you can take a look at PR [#17](https://github.com/thatbrguy/Dehaze-GAN/pull/17) on how to modify the code so that it can work with the higher versions of NumPy.

2. This repository may not work with TensorFlow 2.x out of the box. It also may not work with eager mode for TensorFlow 1.x out of the box. The code was created a few years ago so consider using an older version of TensorFlow 1.x (maybe around 1.4 to 1.9) in the graph execution mode (which is the default mode for TensorFlow 1.x).

## License:
This repository is open source under the MIT clause. Feel free to use it for academic and educational purposes.
