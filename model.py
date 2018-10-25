import os
import time
import cv2
import vgg19
import numpy as np
import tensorflow as tf

from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from operations import TransposeConv, DropOut
from operations import Conv, ReLU, LeakyReLU, AvgPool, BatchNorm

class GAN():

    def __init__(self, args):
        self.num_discriminator_filters = args.D_filters
        self.layers = args.layers
        self.growth_rate = args.growth_rate
        self.gan_wt = args.gan_wt
        self.l1_wt = args.l1_wt
        self.vgg_wt = args.vgg_wt
        self.restore = args.restore
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.model_name = args.model_name
        self.decay = args.decay
        self.save_samples = args.save_samples
        self.sample_image_dir = args.sample_image_dir
        self.A_dir = args.A_dir
        self.B_dir = args.B_dir
        self.custom_data = args.custom_data
        self.val_fraction = args.val_fraction
        self.val_threshold = args.val_threshold
        self.val_frequency = args.val_frequency
        self.logger_frequency = args.logger_frequency
        
        self.EPS = 10e-12
        self.score_best = -1
        self.ckpt_dir = os.path.join(os.getcwd(), self.model_name, 'checkpoint')
        self.tensorboard_dir = os.path.join(os.getcwd(), self.model_name, 'tensorboard')

    def Layer(self, input_):
        """
        This function creates the components inside a composite layer
        of a Dense Block.
        """
        with tf.variable_scope("Composite"):
            next_layer = BatchNorm(input_, isTrain = self.isTrain)
            next_layer = ReLU(next_layer)
            next_layer = Conv(next_layer, kernel_size = 3, stride = 1, output_channels = self.growth_rate)
            next_layer = DropOut(next_layer, isTrain = self.isTrain, rate = 0.2)

            return next_layer

    def TransitionDown(self, input_, name):

        with tf.variable_scope(name):

            reduction = 0.5
            reduced_output_size = int(int(input_.get_shape()[-1]) * reduction)

            next_layer = BatchNorm(input_, isTrain = self.isTrain, decay = self.decay)
            next_layer = Conv(next_layer, kernel_size = 1, stride = 1, output_channels = reduced_output_size)
            next_layer = DropOut(next_layer, isTrain = self.isTrain, rate = 0.2)
            next_layer = AvgPool(next_layer)

            return next_layer

    def TransitionUp(self, input_, output_channels, name):

        with tf.variable_scope(name):
            next_layer = TransposeConv(input_, output_channels = output_channels, kernel_size = 3)
            
            return next_layer

    def DenseBlock(self, input_, name, layers = 4):

        with tf.variable_scope(name):
            for i in range(layers):
                with tf.variable_scope("Layer" + str(i + 1)) as scope:
                    output = self.Layer(input_)
                    output = tf.concat([input_, output], axis=3)
                    input_ = output

        return output

    def generator(self, input_):
        """
        54 Layer Tiramisu
        """
        with tf.variable_scope('InputConv') as scope:
            input_ = Conv(input_, kernel_size = 3, stride=1, output_channels = self.growth_rate * 4)

        collect_conv = []

        for i in range(1, 6):
            input_ = self.DenseBlock(input_, name = 'Encoder' + str(i), layers = self.layers)
            collect_conv.append(input_)
            input_ = self.TransitionDown(input_, name = 'TD' + str(i))

        input_ = self.DenseBlock(input_, name = 'BottleNeck', layers = 15)

        for i in range(1, 6):
            input_ = self.TransitionUp(input_, output_channels = self.growth_rate * 4, name = 'TU' + str(6 - i))
            input_ = tf.concat([input_, collect_conv[6 - i - 1]], axis = 3, name = 'Decoder' + str(6 - i) + '/Concat')
            input_ = self.DenseBlock(input_, name = 'Decoder' + str(6 - i), layers = self.layers)

        with tf.variable_scope('OutputConv') as scope:
            output = Conv(input_, kernel_size = 1, stride = 1, output_channels = 3)

        return tf.nn.tanh(output)

    def discriminator(self, input_, target, stride = 2, layer_count = 4):
        """
        Using the PatchGAN as a discriminator
        """
        input_ = tf.concat([input_, target], axis=3, name='Concat')
        layer_specs = self.num_discriminator_filters * np.array([1, 2, 4, 8])

        for i, output_channels in enumerate(layer_specs, 1):

            with tf.variable_scope('Layer' + str(i)) as scope:
         
                if i != 1:
                    input_ = BatchNorm(input_, isTrain = self.isTrain)
         
                if i == layer_count:
                    stride = 1
         
                input_ = LeakyReLU(input_)
                input_ = Conv(input_, output_channels = output_channels, kernel_size = 4, stride = stride, padding = 'VALID', mode = 'discriminator')

        with tf.variable_scope('Final_Layer') as scope:
            output = Conv(input_, output_channels = 1, kernel_size = 4, stride = 1, padding = 'VALID', mode = 'discriminator')

        return tf.sigmoid(output)

    def build_vgg(self, img):

        model = vgg19.Vgg19()
        img = tf.image.resize_images(img, [224, 224])
        layer = model.feature_map(img)
        return layer

    def build_model(self):

        with tf.variable_scope('Placeholders') as scope:
            self.RealA = tf.placeholder(name='A', shape=[None, 256, 256, 3], dtype=tf.float32)
            self.RealB = tf.placeholder(name='B', shape=[None, 256, 256, 3], dtype=tf.float32)
            self.isTrain = tf.placeholder(name = "isTrain", shape = None, dtype = tf.bool)
            self.step = tf.train.get_or_create_global_step()

        with tf.variable_scope('Generator') as scope:
            self.FakeB = self.generator(self.RealA)

        with tf.name_scope('Real_Discriminator'):
            with tf.variable_scope('Discriminator') as scope:
                self.predict_real = self.discriminator(self.RealA, self.RealB)

        with tf.name_scope('Fake_Discriminator'):
            with tf.variable_scope('Discriminator', reuse=True) as scope:
                self.predict_fake = self.discriminator(self.RealA, self.FakeB)

        with tf.name_scope('Real_VGG'):
            with tf.variable_scope('VGG') as scope:
                self.RealB_VGG = self.build_vgg(self.RealB)

        with tf.name_scope('Fake_VGG'):
            with tf.variable_scope('VGG', reuse=True) as scope:
                self.FakeB_VGG = self.build_vgg(self.FakeB)

        with tf.name_scope('DiscriminatorLoss'):
            self.D_loss = tf.reduce_mean(-(tf.log(self.predict_real + self.EPS) + tf.log(1 - self.predict_fake + self.EPS)))

        with tf.name_scope('GeneratorLoss'):
            self.gan_loss = tf.reduce_mean(-tf.log(self.predict_fake + self.EPS))
            self.l1_loss = tf.reduce_mean(tf.abs(self.RealB - self.FakeB))
            self.vgg_loss = (1e-5) * tf.losses.mean_squared_error(self.RealB_VGG, self.FakeB_VGG)

            self.G_loss = self.gan_wt * self.gan_loss + self.l1_wt * self.l1_loss + self.vgg_wt * self.vgg_loss

        with tf.name_scope('Summary'):
            D_loss_sum = tf.summary.scalar('Discriminator Loss', self.D_loss)
            G_loss_sum = tf.summary.scalar('Generator Loss', self.G_loss)
            gan_loss_sum = tf.summary.scalar('GAN Loss', self.gan_loss)
            l1_loss_sum = tf.summary.scalar('L1 Loss', self.l1_loss)
            vgg_loss_sum = tf.summary.scalar('VGG Loss', self.gan_loss)
            output_img = tf.summary.image('Output', self.FakeB, max_outputs = 1)
            target_img = tf.summary.image('Target', self.RealB, max_outputs = 1)
            input_img = tf.summary.image('Input', self.RealA, max_outputs = 1)

            self.image_summary = tf.summary.merge([output_img, target_img, input_img])
            self.G_summary = tf.summary.merge([gan_loss_sum, l1_loss_sum, vgg_loss_sum, G_loss_sum])
            self.D_summary = D_loss_sum

        with tf.name_scope('Variables'):
            self.G_vars = [var for var in tf.trainable_variables() if var.name.startswith("Generator")]
            self.D_vars = [var for var in tf.trainable_variables() if var.name.startswith("Discriminator")]

        with tf.name_scope('Save'):
            self.saver = tf.train.Saver(max_to_keep=3)

        with tf.name_scope('Optimizer'):

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):

                with tf.name_scope("Discriminator_Train"):
                    D_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)
                    self.D_grads_and_vars = D_optimizer.compute_gradients(self.D_loss, var_list = self.D_vars)
                    self.D_train = D_optimizer.apply_gradients(self.D_grads_and_vars, global_step = self.step)

                with tf.name_scope("Generator_Train"):
                    G_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)
                    self.G_grads_and_vars = G_optimizer.compute_gradients(self.G_loss, var_list = self.G_vars)
                    self.G_train = G_optimizer.apply_gradients(self.G_grads_and_vars, global_step = self.step)

    def train(self):

        start_epoch = 0
        logger_frequency = self.logger_frequency
        val_frequency = self.val_frequency
        val_threshold = self.val_threshold

        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)

        print('Loading Model')
        self.build_model()
        print('Model Loaded')

        print('Loading Data')

        if self.custom_data:

            # Please ensure that the input images and target images have
            # the same filename.

            data = sorted(os.listdir(self.A_dir))

            total_image_count = int(len(data) * (1 - self.val_fraction))
            batches = total_image_count // self.batch_size

            train_data = data[: total_image_count]
            val_data = data[total_image_count: ]
            val_image_count = len(val_data)

            self.A_train = np.zeros((total_image_count, 256, 256, 3))
            self.B_train = np.zeros((total_image_count, 256, 256, 3))
            self.A_val = np.zeros((val_image_count, 256, 256, 3))
            self.B_val = np.zeros((val_image_count, 256, 256, 3))

            print(self.A_train.shape, self.A_val.shape)

            for i, file in enumerate(train_data):
                self.A_train[i] = cv2.imread(os.path.join(os.getcwd(), self.A_dir, file), 1).astype(np.float32)
                self.B_train[i] = cv2.imread(os.path.join(os.getcwd(), self.B_dir, file), 1).astype(np.float32)

            for i, file in enumerate(val_data):
                self.A_val[i] = cv2.imread(os.path.join(os.getcwd(), self.A_dir, file), 1).astype(np.float32)
                self.B_val[i] = cv2.imread(os.path.join(os.getcwd(), self.B_dir, file), 1).astype(np.float32)

        else:
    
            self.A_train = np.load('A_train.npy').astype(np.float32)
            self.B_train = np.load('B_train.npy').astype(np.float32)
            self.A_val = np.load('A_val.npy').astype(np.float32)  # Valset 2
            self.B_val = np.load('B_val.npy').astype(np.float32)

            total_image_count = len(self.A_train)
            val_image_count = len(self.A_val)
            batches = total_image_count // self.batch_size

        self.A_val = (self.A_val / 255) * 2 - 1
        self.B_val = (self.B_val / 255) * 2 - 1
        self.A_train = (self.A_train / 255) * 2 - 1
        self.B_train = (self.B_train / 255) * 2 - 1
    
        print('Data Loaded')
        

        with tf.Session() as self.sess:

            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

            if self.restore:
                print('Loading Checkpoint')
                ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
                self.saver.restore(self.sess, ckpt)
                self.step = tf.train.get_or_create_global_step()
                print('Checkpoint Loaded')

            self.writer = tf.summary.FileWriter(self.tensorboard_dir, tf.get_default_graph())
            total_parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
            G_parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables() if v.name.startswith("Generator")])
            D_parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables() if v.name.startswith("Discriminator")])
            loss_operations = [self.D_loss, self.G_loss, self.gan_loss, self.l1_loss, self.vgg_loss]

            counts = self.sess.run([G_parameter_count, D_parameter_count, total_parameter_count])

            print('Generator parameter count:', counts[0])
            print('Discriminator parameter count:', counts[1])
            print('Total parameter count:', counts[2])

            # The variable below is divided by 2 since both the Generator 
            # and the Discriminator increases step count by 1
            start = self.step.eval() // (batches * 2)

            for i in range(start, self.epochs):

                print('Epoch:', i)
                shuffle = np.random.permutation(total_image_count)

                for j in range(batches):

                    if j != batches - 1:
                        current_batch = shuffle[j * self.batch_size: (j + 1) * self.batch_size]
                    else:
                        current_batch = shuffle[j * self.batch_size: ]

                    a = self.A_train[current_batch]
                    b = self.B_train[current_batch]
                    feed_dict = {self.RealA: a, self.RealB: b, self.isTrain: True}

                    begin = time.time()
                    step = self.step.eval()

                    _, D_summary = self.sess.run([self.D_train, self.D_summary], feed_dict = feed_dict)

                    self.writer.add_summary(D_summary, step)

                    _, G_summary = self.sess.run([self.G_train, self.G_summary], feed_dict = feed_dict)

                    self.writer.add_summary(G_summary, step)

                    print('Time Per Step: ', format(time.time() - begin, '.3f'), end='\r')

                    if j % logger_frequency == 0:
                        D_loss, G_loss, GAN_loss, L1_loss, VGG_loss = self.sess.run(loss_operations, feed_dict=feed_dict)

                        GAN_loss = GAN_loss * self.gan_wt
                        L1_loss = L1_loss * self.l1_wt
                        VGG_loss = VGG_loss * self.vgg_wt

                        trial_image_idx = np.random.randint(total_image_count)
                        a = self.A_train[trial_image_idx]
                        b = self.B_train[trial_image_idx]

                        if a.ndim == 3:
                            a = np.expand_dims(a, axis = 0)

                        if b.ndim == 3:
                            b = np.expand_dims(b, axis = 0)

                        feed_dict = {self.RealA: a, self.RealB: b, self.isTrain: False}
                        img_summary = self.sess.run(self.image_summary, feed_dict=feed_dict)
                        self.writer.add_summary(img_summary, step)

                        line = 'Batch: %d, D_Loss: %.3f, G_Loss: %.3f, GAN: %.3f, L1: %.3f, P: %.3f' % (
                            j, D_loss, G_loss, GAN_loss, L1_loss, VGG_loss)
                        print(line)

                    # The variable `step` counts both D and G updates as individual steps.
                    # The variable `G_D_step` counts one D update followed by a G update
                    # as a single step.
                    G_D_step = step // 2
                    print('GD', G_D_step, 'val', val_threshold)

                    if (val_threshold > G_D_step) and (j % val_frequency == 0):
                        self.validate()


    def validate(self):

        total_ssim = 0
        total_psnr = 0
        psnr_weight = 1/20
        ssim_weight = 1
        val_image_count = len(self.A_val)
 
        for i in range(val_image_count):

            x = np.expand_dims(self.A_val[i], axis = 0)
            feed_dict = {self.RealA: x ,self.isTrain: False}
            generated_B = self.FakeB.eval(feed_dict = feed_dict)

            print('Validation Image', i, end = '\r')

            generated_B = (((generated_B[0] + 1)/2) * 255).astype(np.uint8)
            real_B = (((self.B_val[i] + 1)/2)*255).astype(np.uint8)

            psnr = compare_psnr(real_B, generated_B)
            ssim = compare_ssim(real_B, generated_B, multichannel = True)

            total_psnr = total_psnr + psnr
            total_ssim = total_ssim + ssim

        average_psnr = total_psnr / val_image_count
        average_ssim = total_ssim / val_image_count

        score = average_psnr * psnr_weight + average_ssim * ssim_weight


        if(score > self.score_best):

            self.score_best = score

            self.saver.save(self.sess, os.path.join(self.ckpt_dir, 'gan'), global_step = self.step.eval())
            line = 'Better Score: %.6f, PSNR: %.6f, SSIM: %.6f' %(score, average_psnr, average_ssim)
            print(line)

            with open(os.path.join(self.ckpt_dir, 'logs.txt'),'a') as f:
                line += '\n'
                f.write(line)

            if self.save_samples:

                try:
                    image_list = os.listdir(self.sample_image_dir)
                except:
                    print('Sample images not found. Terminating program')
                    exit(0)
                
                for i, file in enumerate(image_list, 1):
                    
                    print('Sample Image', i, end = '\r')

                    x = cv2.imread(os.path.join(self.sample_image_dir, file), 1)
                    x = (x/255)*2 - 1
                    x = np.reshape(x,(1,256,256,3))

                    feed_dict = {self.RealA: x, self.isTrain: False}
                    img = self.FakeB.eval(feed_dict = feed_dict)

                    img = img[0,:,:,:]
                    img = (((img + 1)/2) * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(self.ckpt_dir, file), img)

    
    def test(self, input_dir, GT_dir):

        total_ssim = 0
        total_psnr = 0
        psnr_weight = 1/20
        ssim_weight = 1

        GT_list = os.listdir(GT_dir)
        input_list = os.listdir(input_dir)

        print('Loading Model')
        self.build_model()
        print('Model Loaded')

        with tf.Session() as self.sess:

            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

            print('Loading Checkpoint')
            ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
            self.saver.restore(self.sess, ckpt)
            self.step = tf.train.get_or_create_global_step()
            print('Checkpoint Loaded')

            for i, (img_file, GT_file) in enumerate(zip(input_list, GT_list), 1):

                img = cv2.imread(os.path.join(input_dir, img_file), 1)
                GT = cv2.imread(os.path.join(GT_dir, GT_file), 1).astype(np.uint8)

                print('Test image', i, end = '\r')

                img = ((np.expand_dims(img, axis = 0) / 255) * 2) - 1
                feed_dict = {self.RealA: img, self.isTrain: False}
                generated_B = self.FakeB.eval(feed_dict = feed_dict)
                generated_B = (((generated_B[0] + 1)/2) * 255).astype(np.uint8)        

                psnr = compare_psnr(GT, generated_B)
                ssim = compare_ssim(GT, generated_B, multichannel = True)

                total_psnr = total_psnr + psnr
                total_ssim = total_ssim + ssim

            average_psnr = total_psnr / len(GT_list)
            average_ssim = total_ssim / len(GT_list)

            score = average_psnr * psnr_weight + average_ssim * ssim_weight

            line = 'Score: %.6f, PSNR: %.6f, SSIM: %.6f' %(score, average_psnr, average_ssim)
            print(line)


    def inference(self, input_dir, result_dir):

        input_list = os.listdir(input_dir)

        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        print('Loading Model')
        self.build_model()
        print('Model Loaded')

        with tf.Session() as self.sess:

            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

            print('Loading Checkpoint')
            ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
            self.saver.restore(self.sess, ckpt)
            self.step = tf.train.get_or_create_global_step()
            print('Checkpoint Loaded')

            for i, img_file in enumerate(input_list, 1):

                img = cv2.imread(os.path.join(input_dir, img_file), 1)

                print('Processing image', i, end = '\r')

                img = ((np.expand_dims(img, axis = 0) / 255) * 2) - 1 
                feed_dict = {self.RealA: img, self.isTrain: False}
                generated_B = self.FakeB.eval(feed_dict = feed_dict)
                generated_B = (((generated_B[0] + 1)/2) * 255).astype(np.uint8)

                cv2.imwrite(os.path.join(result_dir, img_file), generated_B)

            print('Done.')
