import os
import argparse
from model import GAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser()

parser.add_argument("--lr", help="Learning Rate (Default = 0.001)",
                    type = float, default = 0.001)
parser.add_argument("--D_filters", help="Number of filters in the 1st conv layer of the discriminator (Default = 64)",
                    type = int, default = 64)
parser.add_argument("--layers", help="Number of layers per dense block (Default = 4)",
                    type = int, default = 4)
parser.add_argument("--growth_rate", help="Growth Rate of the dense block (Default = 12) ",
                    type = int, default = 12)
parser.add_argument("--gan_wt", help="Weight of the GAN loss factor (Default = 2)",
                    type = float, default = 2)
parser.add_argument("--l1_wt", help="Weight of the L1 loss factor (Default = 100)",
                    type = float, default = 100)
parser.add_argument("--vgg_wt", help="Weight of the VGG loss factor (Default = 10)",
                    type = float, default = 10)
parser.add_argument("--restore", help = "Restore checkpoint for training (Default = False)",
                    type = bool, default = False)
parser.add_argument("--batch_size", help="Set the batch size (Default = 1)",
                    type = int, default = 1)
parser.add_argument("--decay", help="Batchnorm decay (Default = 0.99)",
                    type = float, default = 0.99)
parser.add_argument("--epochs", help = "Epochs (Default = 200)",
                    type = int, default = 200)
parser.add_argument("--model_name", help = "Set a model name",
                    default = 'model')
parser.add_argument("--save_samples", help = "Generate image samples after validation (Default = False)",
                    type = bool, default = False) 
parser.add_argument("--sample_image_dir", help = "Directory containing sample images (Used only if save_samples is True; Default = samples)",
                    default = 'samples')
parser.add_argument("--A_dir", help = "Directory containing the input images for training, testing or inference (Default = A)",
                    default = 'A')
parser.add_argument("--B_dir", help = "Directory containing the target images for training or testing. In inference mode, this is used to store results (Default = B)",
                    default = 'B')
parser.add_argument("--custom_data", help = "Using your own data as input and target (Default = True)",
                    type = bool, default = True)
parser.add_argument("--val_fraction", help = "Fraction of dataset to be split for validation (Default = 0.15)",
                   type = float, default = 0.15)
parser.add_argument("--val_threshold", help = "Number of steps to wait before validation is enabled. (Default = 0)",
                   type = int, default = 0)
parser.add_argument("--val_frequency", help = "Number of batches to wait before perfoming the next validation run (Default = 20)",
                   type = int, default = 20)
parser.add_argument("--logger_frequency", help = "Number of batches to wait before logging the next set of loss values (Default = 20)",
                   type = int, default = 20)
parser.add_argument("--mode", help = "Select between train, test or inference modes",
                    default = 'train', choices = ['train', 'test', 'inference'])

if __name__ == '__main__':

    args = parser.parse_args()
    net = GAN(args)
    if args.mode == 'train':
        net.train()
    if args.mode == 'test':
        net.test(args.A_dir, args.B_dir)
    if args.mode == 'inference':
        net.inference(args.A_dir, args.B_dir)
