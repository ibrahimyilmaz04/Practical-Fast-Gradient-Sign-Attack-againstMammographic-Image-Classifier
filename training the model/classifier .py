from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import importlib
import sys
import time

import matplotlib
import numpy as np
import os.path
import random
import tensorflow as tf

matplotlib.use('Agg')
import json
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


def main(args):

    network = importlib.import_module(args.model_def)

    log_dir = 'training/logs'
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = 'training/models'
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    seed = int(time.time() * 1000) % (10000)
    np.random.seed(seed=seed)
    random.seed(seed)
    train_set = get_dataset(args.data_dir)
    if args.test_dir:
        test_set = get_dataset(args.test_dir)
    nrof_classes = len(train_set)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
    else:
        pretrained_model = model_dir

    print('Pre-trained model: %s' % pretrained_model)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Get a list of image paths and their labels
        image_list, label_list = get_image_paths_and_labels(train_set)
        if args.test_dir:
            test_paths, test_label_list = get_image_paths_and_labels(test_set)
        assert len(image_list) > 0, 'The dataset should not be empty'

        # Create a queue that produces indices into the image_list and label_list
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                                                    shuffle=True, seed=None, capacity=32)

        index_dequeue_op = index_queue.dequeue_many(args.batch_size * args.epoch_size, 'index_dequeue')

        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')

        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=1000000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(1,), (1,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')

        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()

            images = []
            # raw_images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=1)
                image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                image = tf.image.random_flip_left_right(image)
                # pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 1))
                image = tf.cast(image, tf.float32)
                # images.append(tf.image.per_image_standardization(image))
                images.append(image)

            images_and_labels.append([images, label])

        image_batch, label_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 1), ()], enqueue_many=True,
            capacity=8 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')

        print('Total number of classes: %d' % nrof_classes)
        print('Total number of examples: %d' % len(image_list))

        print('Building training graph')

        # Build the inference graph
        prelogits, _ = network.inference(image_batch, args.keep_probability,
                                         phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                                         weight_decay=args.weight_decay)
        logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(args.weight_decay),
                                      scope='Logits', reuse=False)
        proper = tf.nn.softmax(tf.get_default_graph().get_tensor_by_name("Logits/BiasAdd:0"), name='property')
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args.learning_rate_decay_epochs * args.epoch_size,
                                                   args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = trainop(total_loss, global_step, learning_rate, args.moving_average_decay, tf.global_variables(),
                                     args.log_histograms)

        # Create a saver
        # saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        saver = tf.train.Saver()

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=False))
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            if pretrained_model:
                ckpt_state = tf.train.get_checkpoint_state(pretrained_model)
                if ckpt_state and ckpt_state.model_checkpoint_path:
                    print('Restoring pretrained model: %s' % ckpt_state.model_checkpoint_path)
                    saver.restore(sess, ckpt_state.model_checkpoint_path)

            # Training and validation loop
            print('Running training')
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
                      labels_placeholder,
                      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
                      total_loss, train_op, summary_op, summary_writer, regularization_losses)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, '', step)

                # evaluate
                if args.test_dir:
                    evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
                             batch_size_placeholder, proper, label_batch, test_paths, test_label_list, 1, 2, log_dir,
							 step, summary_writer)

    return model_dir

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

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

def _add_loss_summaries(total_loss):

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def trainop(total_loss, global_step, learning_rate, moving_average_decay, update_gradient_vars,
          log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdagradOptimizer(learning_rate)

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def find_threshold(var, percentile):
    var.sort()
    print(type(var[-1]))

    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile * 0.01, cdf, bin_centers)
    return threshold

def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder,
        logits, labels, image_paths, label_list, batch_size, nrof_folds, log_dir, step, summary_writer):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Running forward pass on test images')

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.arange(0,len(image_paths)),1)
    image_paths_array = np.expand_dims(np.array(image_paths),1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    logit_size = logits.get_shape()[1]
    nrof_images = len(image_paths)
    assert nrof_images % batch_size == 0, 'The number of test images must be an integer multiple of the test batch size'
    nrof_batches = nrof_images // batch_size
    logit_array = []
    lab_array = []
    logit_size = logits.get_shape()[1]
    for _ in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        logi, lab,  = sess.run([logits, labels], feed_dict=feed_dict)
        lab = np.array(lab, dtype=np.int)
        lab_array.extend(lab)
        logit_array.extend(logi)
    diff = np.abs(np.argmax(logit_array, axis=1) - label_list)
    diff = np.squeeze(diff)

    err = np.mean(diff)

    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    print('Mean error:%f%%'%(err * 100))
    #pylint: disable=maybe-no-member
    summary.value.add(tag='mean_error', simple_value=err)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'test_result.txt'),'at') as f:
        f.write('%d\t%.5f\n' % (step, err))

def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with open(data_filename, 'r') as f:
        metrics_data = json.load(f)
        distance_to_center = np.array(list(metrics_data['distance_to_center']))
        label_list = metrics_data['label_list']
        image_list = metrics_data['image_list']
        # idx = np.where(distance_to_center == np.nan)
        for idx in range(len(distance_to_center)):
            if distance_to_center[idx] == np.nan:
                print(idx)

        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center >= distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths) < min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del (filtered_dataset[i])

    return filtered_dataset


def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
          loss, train_op, summary_op, summary_writer, regularization_losses):
    batch_number = 0

    lr = args.learning_rate

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True,
                     batch_size_placeholder: args.batch_size}
        if (batch_number % 100 == 0):
            err, _, step, reg_loss, summary_str = sess.run(
                [loss, train_op, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, step, reg_loss = sess.run([loss, train_op, global_step, regularization_losses], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
              (epoch, batch_number + 1, args.epoch_size, duration, err, np.sum(reg_loss)))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing mammographic images for training. Multiple directories are separated with colon.',
                        default='db')
    parser.add_argument('--test_dir', type=str,
                        help='Path to the data directory containing mammographic images for test. Multiple directories are separated with colon.',
                        default='test')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='IRNet')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=100)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=64)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=256)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=100)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0000)
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=10)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=0.1)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of wetraining parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
                        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
