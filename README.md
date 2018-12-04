# Dehaze-GAN
This repository contains TensorFlow code for the paper titled **Single Image Haze Removal using a Generative Adversarial Network.** [[Demo](https://www.youtube.com/watch?v=ioSL6ese46A)][[Arxiv](http://arxiv.org/abs/1810.09479)]

<p align = "center">
	<img src="/src/fog.png" alt="Dehaze-GAN in action" height=50% width=50%>
	</img>
</p>

## Features:
The model has the following components:
- The 56-Layer Tiramisu as the generator.
- A patch-wise discriminator.
- A weighted loss function involving three components, namely:
  - GAN loss component.
  - Perceptual loss component (aka VGG loss component).
  - L1 loss component.

The GAN loss component is dervied from the pix2pix GAN paper. Perceptual loss involves using only the Feature Reconstruction Loss component from this [work](https://arxiv.org/pdf/1603.08155.pdf). 

<p align = "center">
	<img src="/src/model.png" alt="Block diagram of the Dehaze-GAN">
	</img>
</p>

## Notes: 
1. The first version of this project was completed around December 2017. The demo video (dated March 2018) reflects the performance of one of the final versions, however some iterative improvements were made after that. 
2. This repository contains code that can be used for any application, and is not limited to Dehazing. 
3. For recreating the results reported in the paper, use the repository `legacy` (for more details refer below). This repository is the refactored version of the final model, but it uses newer versions of some TensorFlow operations. Those operations are not available in the old saved checkpoints.

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

2. A VGG-19 pretrained on the ImageNet dataset is required to calculate Perceptual loss. In this work, we used the weights provided by [machrisaa](https://github.com/machrisaa/tensorflow-vgg)'s implementation. Download the weights from this [link](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) and include it in this repository.
> **Note:** You can use Keras' pretrained VGG-19 as well, as it can automatically download the ImageNet weights. However, my original implementation did not use it.

3. Download the dataset.
- We used the [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and the [Make 3D](http://make3d.cs.cornell.edu/data.html) dataset for training. The following code will download the NYU Depth Dataset V2 and create the hazy and clear image pairs. The images will be placed in directories `A` and `B` respectively. The formula and values used to create the synthetic haze are stated in our paper.
```
wget -O data.mat http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
python extract.py
```
> **Note:** We were not able to access some ground truth files from the Make3D link and hence we provided steps only for the NYU dataset. However, the algorithm in `extract.py` should apply for any dataset with depth information.

4. In case you want to use your own dataset, follow these instructions. If not, skip this step.
- Create two directories `A` and `B` in this repository. 
- Place the input images into directory `A` and target images into directory `B`. 
- Ensure that an input and target image pair has the **same name**, otherwise the program will throw an error (For instance, if `1.jpg` is present in `A` it must also be present in `B`). 
- Resize all images to be of size `(256, 256, 3)`.

5. Train the model by using the following code. 
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
- `--B_dir`: Directory containing the ipnut images. Only used when `custom_data` is set to `True`. Default value is `B`.
- `--val_fraction`: Fraction of the data to be used for validation. Only used when `custom_data` is set to `True`. Default value is `0.15`.
- `--val_threshold`: Number of steps to wait before validation is enabled. Usually, the GAN performs suboptimally for quite a while. Hence, disabling validation initially can prevent unnecessary validation and speeds up training. Default value is `0`.
- `--val_frequency`: Number of batches to wait before performing the next validation run. Setting this to `1` will perform validation after one discriminator and generator step. You can set it to a higher value to speed up training. Default value is `20`.
- `--logger_frequency`: Number of batches to wait before logging the next set of loss values. Setting it to a higher value will reduce clutter and slightly increase training speed. Default value is `20`.

## Replicating:
The code in `legacy` can be used for replicating results of our model. The model architecture is the same, with the exception of a few old TF operations and messy code. Nevertheless, functionality is the same. The following steps explains how to replicate the results:

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

## License:
This repository is open source under the MIT clause. Feel free to use it for academic and educational purposes.
