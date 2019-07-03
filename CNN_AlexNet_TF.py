import numpy as numpy
import tensorflow as tf
import cv2
import os
from os import listdir
import fnmatch
import progressbar
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
import tools_CNN_view
# ----------------------------------------------------------------------------------------------------------------------
net_data = numpy.load('../_weights/bvlc_alexnet.npy', encoding="latin1",allow_pickle=True).item()
# ----------------------------------------------------------------------------------------------------------------------
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])
# ----------------------------------------------------------------------------------------------------------------------
class CNN_AlexNet_TF():
    def __init__(self):
        self.name = 'CNN_AlexNet_TF'
        self.input_shape = (227, 227)
        self.nb_classes = 4096
        self.x = tf.placeholder(tf.float32, (None, self.input_shape[0], self.input_shape[1], 3))
        self.input_placeholder = tf.image.resize_images(self.x, (227, 227))

        self.class_names = tools_CNN_view.class_names
        self.build()

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def build(self):

        self.conv1W = tf.Variable(net_data["conv1"][0])
        self.conv2W = tf.Variable(net_data["conv2"][0])
        self.conv3W = tf.Variable(net_data["conv3"][0])
        self.conv4W = tf.Variable(net_data["conv4"][0])
        self.conv5W = tf.Variable(net_data["conv5"][0])

        self.conv1Wb = tf.Variable(net_data["conv1"][1])

        self.conv1_in = conv(self.input_placeholder, self.conv1W, tf.Variable(net_data["conv1"][1]), 11, 11, 96, 4, 4, padding="SAME", group=1)
        self.conv1 = tf.nn.relu(self.conv1_in)
        self.lrn1 = tf.nn.local_response_normalization(self.conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
        self.maxpool1 = tf.nn.max_pool(self.lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        self.conv2_in = conv(self.maxpool1, self.conv2W, tf.Variable(net_data["conv2"][1]), 5, 5, 256, 1, 1, padding="SAME", group=2)
        self.conv2 = tf.nn.relu(self.conv2_in)
        self.lrn2 = tf.nn.local_response_normalization(self.conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
        self.maxpool2 = tf.nn.max_pool(self.lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        self.conv3_in = conv(self.maxpool2, self.conv3W, tf.Variable(net_data["conv3"][1]), 3, 3, 384, 1, 1, padding="SAME", group=1)
        self.conv3 = tf.nn.relu(self.conv3_in)
        self.conv4_in = conv(self.conv3, self.conv4W, tf.Variable(net_data["conv4"][1]), 3, 3, 384, 1, 1, padding="SAME", group=2)
        self.conv4 = tf.nn.relu(self.conv4_in)
        self.conv5_in = conv(self.conv4, self.conv5W, tf.Variable(net_data["conv5"][1]), 3, 3, 256, 1, 1, padding="SAME", group=2)
        self.conv5 = tf.nn.relu(self.conv5_in)
        self.maxpool5 = tf.nn.max_pool(self.conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        self.flat5 = tf.reshape(self.maxpool5, [-1, int(numpy.prod(self.maxpool5.get_shape()[1:]))])
        self.fc6 = tf.nn.relu(tf.matmul(self.flat5, tf.Variable(net_data["fc6"][0])) + tf.Variable(net_data["fc6"][1]))
        self.fc7 = tf.nn.relu(tf.matmul(self.fc6, tf.Variable(net_data["fc7"][0])) + tf.Variable(net_data["fc7"][1]))
        self.logits = tf.matmul(self.fc7, tf.Variable(tf.truncated_normal((self.fc7.get_shape().as_list()[-1], self.nb_classes), stddev=1e-2))) + tf.Variable(tf.zeros(self.nb_classes))
        self.layer_feature= tf.nn.softmax(self.logits)
        self.logits = tf.matmul(self.fc7, tf.Variable(net_data["fc8"][0])) + tf.Variable(net_data["fc8"][1])
        self.layer_classes = tf.nn.softmax(self.logits)

        #tf.nn.conv2d_transpose(i, k, [1, s_h, s_w, 1], padding=padding)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def generate_features(self, path_input, path_output,limit=1000000,mask='*.png'):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        if not os.path.exists(path_output):
            os.makedirs(path_output)
        #else:
            #tools_IO.remove_files(path_output)

        patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))

        for each in patterns:
            print(each)
            local_filenames = numpy.array(fnmatch.filter(listdir(path_input + each), mask))[:limit]
            feature_filename = path_output + '/' + each + '.txt'
            features = []

            if not os.path.isfile(feature_filename):
                bar = progressbar.ProgressBar(max_value=len(local_filenames))
                for b, local_filename in enumerate(local_filenames):
                    bar.update(b)
                    image= cv2.imread(path_input + each + '/' + local_filename)
                    image = cv2.resize(image,(self.input_shape[0],self.input_shape[1]))
                    feature = sess.run(self.fc7, feed_dict={self.x: [image]})
                    features.append(feature[0])

                features = numpy.array(features)

                mat = numpy.zeros((features.shape[0], features.shape[1] + 1)).astype(numpy.str)
                mat[:, 0] = local_filenames
                mat[:, 1:] = features
                tools_IO.save_mat(mat, feature_filename, fmt='%s', delim='\t')

        sess.close()
        return
# ---------------------------------------------------------------------------------------------------------------------
    def predict(self,image):

        image_resized = cv2.resize(image, (self.input_shape[0], self.input_shape[1])).astype(numpy.float32)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        prob = sess.run(self.layer_classes, feed_dict={self.x: [image_resized]})[0]
        sess.close()

        idx = numpy.argsort(-prob)[0]
        label = self.class_names[idx]

        return label, prob[idx]
# ---------------------------------------------------------------------------------------------------------------------
    def predict_classes(self, path_input, filename_output, limit=1000000,mask='*.png'):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        #writer = tf.summary.FileWriter('../_images/output/', sess.graph)
        #writer.close()

        patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))

        for each in patterns:
            print(each)
            local_filenames = numpy.array(fnmatch.filter(listdir(path_input + each), mask))[:limit]
            for i in range(0, local_filenames.shape[0]):
                image = cv2.imread(path_input + each + '/' + local_filenames[i])
                image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
                prob = sess.run(self.layer_classes, feed_dict={self.x: [image]})
                prob = prob[0]
                idx = numpy.argsort(-prob)[0]
                label = self.class_names[idx]

                tools_IO.save_labels(path_input+each+'/'+filename_output, numpy.array([local_filenames[i]]), numpy.array([label]), append=i, delim=' ')


        sess.close()
        return
# ---------------------------------------------------------------------------------------------------------------------
    def visualize_filters(self, path_output):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        tensor = self.conv1Wb.eval(sess)


        tensor = 255 * (0.5 + self.conv1W.eval(sess)) #(11,11,3,96)
        cv2.imwrite(path_output + 'filter1.png', tools_CNN_view.tensor_color_4D_to_image(tensor))

        tensor = 255 * (0.5 + self.conv2W.eval(sess)) #(5,5,48,256)
        cv2.imwrite(path_output + 'filter2.png', tools_CNN_view.tensor_gray_4D_to_image(tensor,do_colorize=True))

        tensor = 255 * (0.5 + self.conv3W.eval(sess)) #(3,3,256,384)
        cv2.imwrite(path_output + 'filter3.png', tools_CNN_view.tensor_gray_4D_to_image(tensor,do_colorize=True))

        tensor = 255 * (0.5 + self.conv4W.eval(sess)) #(3,3,192,384)
        cv2.imwrite(path_output + 'filter4.png', tools_CNN_view.tensor_gray_4D_to_image(tensor,do_colorize=True))

        tensor = 255 * (0.5 + self.conv5W.eval(sess)) #(3,3,192,256)
        cv2.imwrite(path_output + 'filter5.png', tools_CNN_view.tensor_gray_4D_to_image(tensor,do_colorize=True))
        sess.close()
        return
# ---------------------------------------------------------------------------------------------------------------------
    def visualize_layers(self, filename_input, path_output):

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        image = cv2.resize(cv2.imread(filename_input), (self.input_shape[0], self.input_shape[1]))

        output = sess.run(self.conv1, feed_dict={self.x: [image]})[0] #(57,57,96)
        cv2.imwrite(path_output + 'layer01_conv1.png',tools_CNN_view.tensor_gray_3D_to_image(output,do_colorize=True))

        output = sess.run(self.maxpool1, feed_dict={self.x: [image]})[0] #(28,28,96)
        cv2.imwrite(path_output + 'layer02_pool1.png', tools_CNN_view.tensor_gray_3D_to_image(2*output,do_colorize=True))

        output = sess.run(self.conv2, feed_dict={self.x: [image]})[0] #(28,28,256)
        cv2.imwrite(path_output + 'layer03_conv2.png', tools_CNN_view.tensor_gray_3D_to_image(output,do_colorize=True))

        output = sess.run(self.maxpool2, feed_dict={self.x: [image]})[0] #(13,13,256)
        cv2.imwrite(path_output + 'layer04_pool2.png', tools_CNN_view.tensor_gray_3D_to_image(output,do_colorize=True))

        output = sess.run(self.conv3, feed_dict={self.x: [image]})[0]  #(13,13,384)
        cv2.imwrite(path_output + 'layer04_conv3.png', tools_CNN_view.tensor_gray_3D_to_image(output,do_colorize=True))

        output = sess.run(self.conv4, feed_dict={self.x: [image]})[0] #(13,13,384)
        cv2.imwrite(path_output + 'layer05_conv4.png', tools_CNN_view.tensor_gray_3D_to_image(output,do_colorize=True))

        output = sess.run(self.conv5, feed_dict={self.x: [image]})[0] #(13,13,256)
        cv2.imwrite(path_output + 'layer06_conv5.png', tools_CNN_view.tensor_gray_3D_to_image(output,do_colorize=True))

        output = sess.run(self.maxpool5, feed_dict={self.x: [image]})[0] #(6,6,256)
        cv2.imwrite(path_output + 'layer07_pool5.png', tools_CNN_view.tensor_gray_3D_to_image(2*output,do_colorize=True))

        output = sess.run(self.fc6, feed_dict={self.x: [image]})[0] #4096
        cv2.imwrite(path_output + 'layer08_fc6.png', tools_image.hitmap2d_to_viridis(tools_CNN_view.tensor_gray_1D_to_image(output)))

        output = sess.run(self.fc7, feed_dict={self.x: [image]})[0] #4096
        output*=255.0/output.max()
        cv2.imwrite(path_output + 'layer09_features.png', tools_image.hitmap2d_to_viridis(tools_CNN_view.tensor_gray_1D_to_image(output)))

        output = 100*sess.run(self.layer_feature, feed_dict={self.x: [image]})[0] #4096
        cv2.imwrite(path_output + 'layer10.png', tools_image.hitmap2d_to_viridis(tools_CNN_view.tensor_gray_1D_to_image(output)))

        output = 100*sess.run(self.layer_classes, feed_dict={self.x: [image]})[0] #1000
        cv2.imwrite(path_output + 'layer11_classes.png', tools_image.hitmap2d_to_viridis(tools_CNN_view.tensor_gray_1D_to_image(output)))

        idx = numpy.argsort(-output)
        mat = numpy.array([output[idx],numpy.array(self.class_names)[idx]]).T
        tools_IO.save_mat(mat,path_output+'predictions.txt',fmt='%s',delim='\t')

        sess.close()
        return
# ---------------------------------------------------------------------------------------------------------------------
