from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
from tensorflow.python.ops import data_flow_ops
import re
import cv2
# from classifier import get_dataset
# from classifier import get_image_paths_and_labels
import matplotlib.pyplot as plt
from scipy import misc
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

def get_dataset(paths):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            image_paths = get_image_paths(facedir)
            dataset.append(ImageClass(class_name, image_paths))

    return dataset

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def main(args):
	with tf.Graph().as_default():
		with tf.Session() as sess:

			# Load the model
			load_model(args.model)

			# Get input and output tensors
			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			logit = tf.get_default_graph().get_tensor_by_name("Logits/BiasAdd:0")
			batch_size_placeholder = tf.get_default_graph().get_tensor_by_name("batch_size:0")
			labels_placeholder = tf.get_default_graph().get_tensor_by_name("labels:0")
			image_paths_placeholder = tf.get_default_graph().get_tensor_by_name("image_paths:0")
			proper = tf.nn.softmax(logit)
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

			epsilon_placeholder = tf.placeholder(tf.float32, name='epsilon')
			atkimg = step_fgsm(images_placeholder, epsilon_placeholder, proper)

			# Run forward pass to calculate logit
			epsilons = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
			accuracies = []

			for epsilon in epsilons:
				if not epsilon == 0:
					os.makedirs('attack_%02d'%(epsilon*100), exist_ok=True)
				for dir in os.listdir(args.test_db):
					if epsilon == 0:
						continue
					dir_path = os.path.join(args.test_db, dir)
					outdir_path = os.path.join('attack_%02d'%(epsilon*100), dir)
					os.makedirs(outdir_path, exist_ok=True)
					for file in os.listdir(dir_path):
						if not file.lower().endswith('jpg') and not file.lower().endswith('png'):
							continue
						image = cv2.imread(os.path.join(dir_path, file), cv2.IMREAD_GRAYSCALE)
						image = cv2.resize(image, (args.image_size, args.image_size))
						image = np.expand_dims(image, axis=2)
						image = np.expand_dims(image, axis=0)

						# generate attack result
						img1 = sess.run(atkimg, feed_dict={images_placeholder: image, epsilon_placeholder: epsilon,
														   phase_train_placeholder: False})
						img = np.array(img1, np.uint8)
						img = np.squeeze(img)
						cv2.imwrite(os.path.join(outdir_path, file), img)

				# calculate accuracy on attacked images
				if not epsilon == 0:
					test_dir = 'attack_%02d'%(epsilon*100)
				else:
					test_dir = args.test_db
				# test_dir = 'db'
				test_set = get_dataset(test_dir)
				test_paths, test_label_list = get_image_paths_and_labels(test_set)

				logit_array = []
				accuracy = 0
				for i in range(len(test_paths)):

					images = cv2.imread(test_paths[i], cv2.IMREAD_GRAYSCALE)
					images = np.expand_dims(images, axis=2)
					images = np.expand_dims(images, axis=0)
					feed_dict = { images_placeholder:images, phase_train_placeholder:False }
					# feed_dict = {images_placeholder: images}
					logit_ = sess.run(logit, feed_dict=feed_dict)
					if np.argmax(logit_) == test_label_list[i]:
						accuracy += 1


				accuracies.append(accuracy / len(test_paths))

			plt.figure()
			plt.xlabel('epsilon')
			plt.ylabel('accuracy')
			plt.plot(epsilons, accuracies)
			plt.savefig('curve.jpg')

def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 1))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        images[i,:,:,:] = img
    return images

def load_model(model):
	model_exp = os.path.expanduser(model)
	print('Model directory: %s' % model_exp)
	meta_file, ckpt_file = get_model_filenames(model_exp)

	print('Metagraph file: %s' % meta_file)
	print('Checkpoint file: %s' % ckpt_file)

	saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
	saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
	files = os.listdir(model_dir)
	meta_files = [s for s in files if s.endswith('.meta')]
	if len(meta_files) == 0:
		raise ValueError('No meta file found in the model directory (%s)' % model_dir)
	elif len(meta_files) > 1:
		raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
	meta_file = meta_files[0]
	meta_files = [s for s in files if '.ckpt' in s]
	max_step = -1
	for f in files:
		step_str = re.match(r'(^model-+.ckpt-(\d+))', f)
		if step_str is not None and len(step_str.groups()) >= 2:
			step = int(step_str.groups()[1])
			if step > max_step:
				max_step = step
				ckpt_file = step_str.groups()[0]
	return meta_file, ckpt_file


def step_fgsm(x, eps, logits):
	label = tf.argmax(logits, 1)
	one_hot_label = tf.one_hot(label, 2)
	cross_entropy = tf.losses.softmax_cross_entropy(one_hot_label,
													logits,
													label_smoothing=0.1,
													weights=1.0)
	x_adv = x + eps * (tf.sign(tf.gradients(cross_entropy, x)[0]) * 128 + 127.5)
	# x_adv = x + eps * (tf.sign(tf.gradients(cross_entropy, x)[0]) * 256)
	x_adv = tf.clip_by_value(x_adv,0,255)

	return tf.stop_gradient(x_adv)

def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('--test_db', type=str, default='train_db',
						help='Path of image to test.')
	parser.add_argument('--model', type=str, default='training/models',
						help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
	parser.add_argument('--image_size', type=int,
						help='Image size (height, width) in pixels.', default=256)
	return parser.parse_args(argv)


if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
