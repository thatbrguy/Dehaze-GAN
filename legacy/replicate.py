import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr
import vgg19small
from utils import *

class GAN():

    def __init__(self):
        self.gen_filters = 64
        self.disc_filters = 64
        self.layers = 4
        self.growth_rate = 12
        self.gan_wt = 2
        self.l1_wt = 100
        self.vgg_wt = 10
        self.num = 14
        self.restore = True
        self.ckpt_dir = './model/'+str(self.num)+'/checkpoint'
        self.batch_sz = 1
        self.epochs = 1000
        self.lr = 0.001
        self.total_image_count = 1550 * 2 #Due to flips
        self.score_best = -1


    def Layer(self, ip):

        with tf.variable_scope("Composite"):
            next_layer = batchnorm(ip)
            next_layer = Relu(next_layer)
            next_layer = Conv(next_layer, filter = 3, stride = 1, output_ch = self.growth_rate)
            next_layer = DropOut(next_layer, rate=0.2)
            
            return next_layer

    def TransitionDown(self, ip, name):

        with tf.variable_scope(name):

            reduction = 0.5
            next_layer = batchnorm(ip)
            reduced_output_size  = int(int(ip.get_shape()[-1]) * reduction)
            next_layer = Conv(next_layer, filter = 1, stride =1, output_ch = reduced_output_size) 
            next_layer = DropOut(next_layer, rate=0.2)
            next_layer = AvgPool(next_layer)

            return next_layer

    def TransitionUp(self, ip, output_ch, name):

        with tf.variable_scope(name):
            next_layer = deconv(ip, output_ch, filters = 3)
            return next_layer

    def DenseBlock(self, ip, name, layers = 4):

        with tf.variable_scope(name):
            for i in range(layers):
                with tf.variable_scope("Layer" + str(i+1)) as scope:
                    output = self.Layer(ip)
                    output = tf.concat([ip, output], axis = 3)
                    ip = output

        return output

    def tiramisu(self, ip):

        with tf.variable_scope('InputConv') as scope:
            ip = Conv(ip, filter = 3, stride = 1, output_ch = self.growth_rate*4)

        collect_conv = []

        for i in range(1,6):
            ip = self.DenseBlock(ip, 'Encoder' + str(i), layers = self.layers)
            collect_conv.append(ip)
            ip = self.TransitionDown(ip, 'TD' + str(i))

        ip = self.DenseBlock(ip, 'BottleNeck', layers = 15)

        for i in range(1,6):
            ip = self.TransitionUp(ip, self.growth_rate*4, 'TU' + str(6 - i))
            ip = tf.concat([ip, collect_conv[6-i-1]], axis = 3, name = 'Decoder' + str(6-i) + '/Concat')
            ip = self.DenseBlock(ip, 'Decoder' + str(6 - i), layers = self.layers)

        with tf.variable_scope('OutputConv') as scope:
            output = Conv(ip, filter = 1, stride = 1, output_ch = 3)
        
        return tf.nn.tanh(output) 


    def discriminator(self, ip, target):

        #Using the PatchGAN as a discriminator
        layer_count = 4
        stride = 2
        ndf = self.disc_filters
        ip = tf.concat([ip, target], axis = 3, name = 'Concat')

        layer_specs = ndf * np.array([1,2,4,8])

        for i, out_ch in enumerate(layer_specs,1):

            with tf.variable_scope('Layer'+str(i)) as scope:
                if i != 1:
                    ip = batchnorm(ip)
                ip = lrelu(ip)
                if i == layer_count:
                    stride = 1
                ip = conv(ip, out_ch, stride = stride)

        with tf.variable_scope('Final_Layer') as scope:
            ip = conv(ip, out_channels = 1, stride = 1)
            output = tf.sigmoid(ip)

        return output

    def build_vgg(self, img):

        model = vgg19small.Vgg19()
        img = tf.image.resize_images(img, [224,224])
        layer = model.feature_map(img)
        return layer


    def build(self):

        EPS = 10e-12

        with tf.variable_scope('Placeholders') as scope:
            self.RealA = tf.placeholder(name = 'A', shape = [None, 256, 256, 3], dtype = tf.float32)
            self.RealB = tf.placeholder(name = 'B', shape = [None, 256, 256, 3], dtype = tf.float32)
            self.step = tf.train.get_or_create_global_step()

        with tf.variable_scope('Generator') as scope:
            self.FakeB = self.tiramisu(self.RealA)

        with tf.name_scope('Real_Discriminator'):
            with tf.variable_scope('Discriminator') as scope:
                self.predict_real = self.discriminator(self.RealA, self.RealB)

        with tf.name_scope('Fake_Discriminator'):
            with tf.variable_scope('Discriminator', reuse = True) as scope:
                self.predict_fake = self.discriminator(self.RealA, self.FakeB)

        with tf.name_scope('Real_VGG'):
            with tf.variable_scope('VGG') as scope:
                self.RealB_VGG = self.build_vgg(self.RealB)

        with tf.name_scope('Fake_VGG'):
            with tf.variable_scope('VGG', reuse = True) as scope:
                self.FakeB_VGG = self.build_vgg(self.FakeB)

        with tf.name_scope('DiscriminatorLoss'):
            self.D_loss = tf.reduce_mean(-( tf.log(self.predict_real + EPS) + tf.log(1 - self.predict_fake + EPS) ))

        with tf.name_scope('GeneratorLoss'):
            self.gan_loss = tf.reduce_mean(-tf.log(self.predict_fake + EPS ))
            self.l1_loss = tf.reduce_mean(tf.abs(self.RealB - self.FakeB))
            self.vgg_loss = (1e-5) * tf.losses.mean_squared_error(self.RealB_VGG, self.FakeB_VGG)

            self.G_loss = self.gan_wt * self.gan_loss + self.l1_wt * self.l1_loss + self.vgg_wt * self.vgg_loss

        with tf.name_scope('Summary'):
            dloss_sum = tf.summary.scalar('Discriminator Loss', self.D_loss)
            gloss_sum = tf.summary.scalar('Generator Loss', self.G_loss)
            gan_loss_sum = tf.summary.scalar('GAN Loss', self.gan_loss)
            l1_loss_sum = tf.summary.scalar('L1 Loss', self.l1_loss)
            vgg_loss_sum = tf.summary.scalar('VGG Loss', self.gan_loss)
            output_im = tf.summary.image('Output', self.FakeB, max_outputs = 1)
            target_im = tf.summary.image('Target', self.RealB, max_outputs = 1)
            input_im = tf.summary.image('Input', self.RealA, max_outputs = 1)

            self.image_summary = tf.summary.merge([output_im, target_im, input_im])
            self.g_summary = tf.summary.merge([gan_loss_sum, l1_loss_sum, vgg_loss_sum, gloss_sum])
            self.d_summary = dloss_sum

        with tf.name_scope('Variables'):
            self.G_vars = [var for var in tf.trainable_variables() if var.name.startswith("Generator")]
            self.D_vars = [var for var in tf.trainable_variables() if var.name.startswith("Discriminator")]

        with tf.name_scope('Save'):
            self.saver = tf.train.Saver(max_to_keep = 10)

        with tf.name_scope('Optimizer'):
            with tf.name_scope("Discriminator_Train"):
                discrim_optim = tf.train.AdamOptimizer(self.lr, beta1 = 0.5)
                self.discrim_grads_and_vars = discrim_optim.compute_gradients(self.D_loss, var_list = self.D_vars)
                self.discrim_train = discrim_optim.apply_gradients(self.discrim_grads_and_vars, global_step = self.step)

            with tf.name_scope("Generator_Train"):
                gen_optim = tf.train.AdamOptimizer(self.lr, beta1 = 0.5)
                self.gen_grads_and_vars = gen_optim.compute_gradients(self.G_loss, var_list = self.G_vars)
                self.gen_train = gen_optim.apply_gradients(self.gen_grads_and_vars, global_step = self.step)


    def test(self, ckpt_dir):

        # Weight values as used in the paper.
        total_ssim = 0
        total_psnr = 0
        psnr_weight = 1/20
        ssim_weight = 1

        self.A_test = np.load('A_test.npy') #Valset 2
        self.B_test = np.load('B_test.npy')
        self.A_test = (self.A_test/255)*2 - 1

        print('Building Model')
        self.build()
        print('Model Built')

        with tf.Session() as sess:

            print('Loading Checkpoint')
            self.ckpt = tf.train.latest_checkpoint(ckpt_dir, latest_filename=None)
            self.saver.restore(sess, self.ckpt)
            print('Checkpoint Loaded')

            for i in range(len(self.A_test)):

                x = np.expand_dims(self.A_test[i], axis = 0)
                feed = {self.RealA :x}
                img = self.FakeB.eval(feed_dict = feed)

                print('Test image', i, end = '\r')

                A_img = (((img[0] + 1)/2) * 255).astype(np.uint8)
                B_img = (self.B_test[i]).astype(np.uint8)

                psnr = compare_psnr(B_img, A_img)
                s = ssim(B_img, A_img, multichannel = True)

                total_psnr = total_psnr + psnr
                total_ssim = total_ssim + s

            average_psnr = total_psnr / len(self.A_test)
            average_ssim = total_ssim / len(self.A_test)

            score = average_psnr * psnr_weight + average_ssim * ssim_weight

            line = 'Score: %.6f, PSNR: %.6f, SSIM: %.6f' %(score, average_psnr, average_ssim)
            print(line)


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    obj = GAN()
    obj.test(os.path.join(os.getcwd(), 'model', 'checkpoint'))
