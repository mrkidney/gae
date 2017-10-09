import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

class OptimizerVAE(object):
    def __init__(self, recons, recon_labels, preds, labels, model, num_nodes):

        mask = tf.reshape(1 - tf.eye(num_nodes, dtype = tf.int32), [-1])
        recons = tf.reshape(recons, [-1, num_nodes*num_nodes])
        recons = tf.gather(recons, mask, axis = 1)
        recon_labels = tf.reshape(recon_labels, [-1, num_nodes*num_nodes])
        recon_labels = tf.gather(recon_labels, mask, axis = 1)

        self.recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=recons, labels=recon_labels), 1)
        self.recon_loss = tf.reduce_mean(self.recon_loss)

        # self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
        #                                                            tf.square(tf.exp(model.z_log_std)), 1))

        self.kl = 0.5 * tf.reduce_sum(tf.exp(model.z_log_std) + tf.square(model.z_mean) - 1.0 - model.z_log_std, [1, 2])
        self.kl = tf.reduce_mean(self.kl)

        preds = tf.gather(preds, model.mask)
        labels = tf.gather(labels, model.mask)
        self.classification_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels))

        self.cost = self.recon_loss
        self.cost += self.kl
        if not FLAGS.vae:
            self.cost = 0
        self.cost += FLAGS.loss_scalar * self.classification_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_reconstruction = tf.equal(tf.cast(tf.greater_equal(recons, 0.5), tf.int32),
                                           tf.cast(recon_labels, tf.int32))
        self.recon_accuracy = tf.reduce_mean(tf.cast(self.correct_reconstruction, tf.float32))

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds, 0.5), tf.int32),
                                           tf.cast(labels, tf.int32))
        self.pred_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
