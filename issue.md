# June 2021 Update

On a recent review of some code I found a few issues. In this document, I provide an explanation of the issues, their possible impact, and the remedies I have taken to account for the issues.

This codebase was created a few years ago when I was an undergraduate student and some of the files are cleaned up versions of the ones used for my experiments. Unfortunately since I was less experienced and messy back then, these issues crept in and it was hard to analyze if something went wrong and if so what are the causes and effects. I have documented and analyzed these issues and suggested remedies based on my recent review. I apologize for the inconvenience.

## Issues

### 1. Colorspace Issue

In the codebase, the file `model.py` loads images using `cv2.imread`. This function loads the image to a NumPy array with the channels in the BGR format. Hence, the model would be trained and tested on BGR images. For ease of training and testing for my experiments, I used to store the images in NumPy files. This was done so that I can simply load the NumPy files instead of reading the images using OpenCV (you might have seen mentions of `A_train.npy`, `B_train.npy`, `A_test.npy` etc. in the codebase). The NumPy files had images in the BGR format and hence the model would be trained and tested on BGR images. Also, to save the model output, I used `cv2.imwrite`, which expects a NumPy array with the channels in BGR format.

I recently realized that the file `extract.py` has a colorspace issue while saving the images to files. I observed that I used `cv2.imwrite` in `extract.py`, and that I passed NumPy arrays with channels in the RGB format to that function in `extract.py`. This is a mistake; I should have passed NumPy arrays with channels in BGR format  to `cv2.imwrite` so that the files are saved in the correct colorspace. 

I have made a change in `extract.py` to fix this issue. Now, the images are converted to the BGR format in `extract.py` before they are saved.

This issue **does not** affect `replicate.py` since it directly reads `A_test.npy` and  `B_test.npy`. As mentioned before, both NumPy files have images already stored in the BGR format, which is then consumed by the model.

### 2. VGG Issue

As mentioned in `README.md`, I used the pre-trained VGG 19 model from [machrisaa](https://github.com/machrisaa/tensorflow-vgg)'s implementation. This model is used for calculating the perceptual loss. On a recent review of the function `feature_map` in the files `vgg19.py` and `legacy/vgg19small.py`, I noticed that the docstring mentioned the following:

```
load variable from npy to build the VGG

:param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
```

Unfortunately, for the final experiments that I had run, I might have forgotten to take into account the above docstring. 

Based on the docstring, it seems like the function needed an input with the following characteristics:

- Image input in the RGB format with the values in the range `[0, 1]`

However, the codebase provided the function an input with the following characteristics:

- Image input in the BGR format with the values in the range `[-1, 1]`

Hence, the VGG model was not used as it ideally should have been used. 

It is interesting to note that we still achieved good performance even though the input to the VGG model did not meet the requirements of the docstring. To think about why and the potential impact, let us analyze the purpose of using this VGG model. We used the pre-trained VGG model to extract features from images. These features are then used in the computation of the perceptual loss. The VGG is only required for training and is not needed for testing/inference.

We can now state a hypothesis. Even in the incorrect setting, we are still computing the perceptual loss between the extracted features of a ground truth image and the extracted features of the corresponding generated image. This anyway motivates the model to generate images such that the features extracted from these generated images are similar to the features extracted from the ground truth images. Hence, the perceptual loss even in the incorrect case still likely serves a purpose. After all, we were able to train the model using the incorrect setting, and then also get good performance during testing. 

However, it is also possible that providing input in the correct colorspace format and with values in the correct range could give better results since the features may have more "useful information". But in both the incorrect and correct settings, we are still motivating the model to generate images such that the features extracted from these generated images are similar to the features extracted from the ground truth images.

However, do note that the above points are just hypotheses. A proper comparision experiment should be done to analyze the impact accurately.

I apologize for the inconvenience cause by these mistakes. The next section discusses what remedies are done in light of these issues.

## Remedies

In light of these issues, a few changes were made to the codebase. I was quite torn on what was the right way to remedy these issues. On one hand, I wanted to preserve the codebase to be as consistent with the paper as possible. On the other hand, I wanted to fix the issues as well. 

After much thought, I have decided to create **two versions**. Each version will lie on a different **branch**. In the `master` branch, the version that is more consistent with the paper is kept. In the `alternate` branch, the version that has resolved an additional issue is kept. The specifics of both branches are given below:

- The `master` branch:
  - Colorspace issue in `extract.py` is fixed.
- The `alt` branch:
  - Colorspace issue in `extract.py` is fixed.
  - The function `feature_map` in `vgg.py` is modified such that it can accept BGR images with values in the range `[-1, 1]`.

The reaons why the `master` branch did not change the `feature_map` function was because the experiments for the paper likely used the function with incorrect input characteristics (see the 2nd issue in the issues section). Hence, to be consistent with the paper, the mistake is left as such without correction in the `master` branch alone. If you would like to use the function **without** the mistake, please use the `alt` branch. However do note that I have **not tested** the code in the `alt` branch and hence cannot provide performance comparisons with the `master` branch. 

The entire `legacy` folder in **both branches** have **no changes made**. This is because the `legacy` folder is used for replicating the paper results and it is being preserved as such (even though it has the same VGG issue).

## Step 2 from README Instructions

Step 2 of the instructions section in the README file asks the user to choose one of the two branches (among `master` or `alt`) for their use case. If you have reached this document from step 2 of the README instructions, please read the rest of this document to understand the differences between the two branches. If you have already read the rest of the document, follow the below instructions to complete step 2 of the instructions section in the README file:

- Make sure that step 1 of instructions in the README file has been completed (i.e. git cloning the repository).

- Now, **if** you would like to use the `master` branch: 

	- Verify that you are in the `master` branch by running `git branch`.
	- Verify that you are **either** using the commit `07bd52840574d0c8e1f3a0482538a674a64618c4` **or** any commit made **after** that by running `git log`. The commit message of commit `07bd52840574d0c8e1f3a0482538a674a64618c4` should say `june common patch` and it should have the month and year as `June 2021`. For simplicity, you can just use the newest commit in the branch (since the newest commit is made after the commit `07bd52840574d0c8e1f3a0482538a674a64618c4`).
	- Once both are verified, you can proceed to step 3 of the instructions in the README file.

- Else, **if** you would like to use the `alt` branch: 

	- First, execute the below commands:
	```
	git fetch origin alt:alt
	git checkout alt
	```
	- Verify that you are in the `alt` branch by running `git branch`.
	- Verify that you are **either** using the commit `95c4e96635e6954499b9c27693a3e11f45019995` **or** any commit made **after** that by running `git log`. The commit message of commit `95c4e96635e6954499b9c27693a3e11f45019995` should say `vgg patch` and it should have the month and year as `June 2021`. For simplicity, you can just use the newest commit in the branch (since the newest commit is made after the commit `95c4e96635e6954499b9c27693a3e11f45019995`).
	- Once both are verified, you can proceed to step 3 of the instructions in the README file.


