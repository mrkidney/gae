from __future__ import division
from __future__ import print_function

import time
import os


import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from optimizer import OptimizerVAE
from input_data import load_mutag
from model import GCNModelAE, GCNModelVAE, MyModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, get_split, apply_indices


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.004, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 10, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 10, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 10, 'Number of units in hidden layer 4.')
flags.DEFINE_float('weight_decay', 5e-12, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('validation', 0.2, 'fraction of data to keep for validation')
flags.DEFINE_float('loss_scalar', 1.0, 'scalar to control classification loss relative to reconstruction loss')
flags.DEFINE_boolean('vae', True, 'setting to false removes reconstruction and KL loss from the objective function')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('gpu', -1, 'Which gpu to use (-1 means using cpu)')
flags.DEFINE_boolean('noisy', True, 'Whether to output results on every epoch')


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
if FLAGS.gpu == -1:
    sess = tf.Session()
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu) # Or whichever device you would like to use
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

model_str = FLAGS.model
dataset_str = FLAGS.dataset



# Load data
adj_orig, features, labels, non_identity_features = load_mutag()
num_examples, num_nodes, num_features = features.shape

adj = np.zeros_like(adj_orig)
for i in range(num_examples):
    adj[i] = preprocess_graph(adj_orig[i]).todense()
    adj_orig[adj_orig != 0] = 1

placeholders = {
    'features': tf.placeholder(tf.float32, shape = (None, num_nodes, num_features)),
    'adj': tf.placeholder(tf.float32, shape = (None, num_nodes, num_nodes)),
    'adj_orig': tf.placeholder(tf.float32, shape = (None, num_nodes, num_nodes)),
    'labels' : tf.placeholder(tf.float32, name = "labels"),
    'mask' : tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

# Create model
model = None
if model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes)
elif model_str == 'my_vae':
    model = MyModelVAE(placeholders, num_features, num_nodes)


# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_vae' or model_str == 'my_vae':
        opt = OptimizerVAE(recons=model.reconstructions, recon_labels=placeholders['adj_orig'], preds=model.preds, 
                           labels=placeholders['labels'], model=model, num_nodes=num_nodes)

def evaluate(adj, adj_orig, features, labels, placeholders, indices, training):
    feed_dict = construct_feed_dict(adj, adj_orig, features, labels, placeholders, indices)
    funcs = [opt.pred_accuracy]
    if training:
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        funcs = [opt.opt_op, opt.cost, opt.recon_accuracy, opt.pred_accuracy, opt.recon_loss, opt.kl]
    outs = sess.run(funcs, feed_dict=feed_dict)
    return outs

# Init variables
sess.run(tf.global_variables_initializer())

cost_train = []
cost_val = []
train_indices, val_indices = get_split(num_examples, FLAGS.validation)

for epoch in range(FLAGS.epochs):

    outs_val = evaluate(adj, adj_orig, features, labels, placeholders, val_indices, training = False)
    cost_val.append(outs_val)
    outs_train = evaluate(adj, adj_orig, features, labels, placeholders, train_indices, training = True)
    cost_train.append(outs_train)

    perm = np.random.permutation(num_nodes)
    adj = adj[:, perm, :]
    adj = adj[:, :, perm]
    adj_orig = adj_orig[:, perm, :]
    adj_orig = adj_orig[:, :, perm]
    features[:, :, non_identity_features:] = features[:, perm, non_identity_features:]

    print("Epoch:", '%04d' % (epoch + 1), "recon_loss=", "{:.5f}".format(outs_train[4]),
                                          "recon_acc=", "{:.5f}".format(outs_train[2]),
                                          "KL=", "{:.5f}".format(outs_train[5]),
                                          "train_acc=", "{:.5f}".format(outs_train[3]),
                                          "val_acc=", "{:.5f}".format(outs_val[0]))

print("Optimization Finished!")
