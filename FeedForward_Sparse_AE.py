# Feed Forward Sparse Autoencoder
import tensorflow as tf
import math
import keras
import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from keras.layers import Input,Dense
from keras import regularizers
from keras.models import model_from_json
from skimage.util import random_noise


def status(X,Y):
        diff = Y-X
        print('loss :'+str(0.5*tf.reduce_mean(tf.reduce_sum(diff**2,axis=1))))

class FeedforwardSparseAutoEncoder():

    def __init__(self, n_input, n_hidden,  rho=0.01, alpha=0.0001, beta=3, activation=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer()):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.rho=rho  # sparse parameters
        self.alpha = alpha
        self.beta=beta
        self.optimizer=optimizer
        self.activation = activation

        self.W1=self.init_weights((self.n_input,self.n_hidden))
        self.b1=self.init_weights((1,self.n_hidden))

        self.W2=self.init_weights((self.n_hidden,self.n_input))
        self.b2= self.init_weights((1,self.n_input))
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def init_weights(self,shape):
        r= math.sqrt(6) / math.sqrt(self.n_input + self.n_hidden + 1)
        weights = tf.random_normal(shape, stddev=r,seed=0)
        return tf.Variable(weights)

    def encode(self,X):
        l=tf.matmul(X, self.W1)+self.b1
        return self.activation(l)

    def decode(self,H):
        l=tf.matmul(H,self.W2)+self.b2
        return self.activation(l)


    def kl_divergence(self, rho, rho_hat):
        return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)

    def regularization(self,weights):
        return tf.nn.l2_loss(weights)

    def loss(self,X,Y):
        H = self.encode(X)
        rho_hat=tf.reduce_mean(H,axis=0)   #Average hidden layer over all data points in X
        kl=self.kl_divergence(self.rho, rho_hat)
        X_=self.decode(H)
        diff=Y-X_
        cost= 0.5*tf.reduce_mean(tf.reduce_sum(diff**2,axis=1))  \
                  +0.5*self.alpha*(tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2))   \
                  +self.beta*tf.reduce_sum(kl)
        return cost
    
    def training(self,training_data,output_data,  n_iter=30):

        X=tf.placeholder("float",shape=[None,training_data.shape[1]])
        Y=tf.placeholder("float",shape=[None,output_data.shape[1]])
        var_list=[self.W1,self.W2]
        loss_=self.loss(X,Y)
        train_step=tf.contrib.opt.ScipyOptimizerInterface(loss_, var_list=var_list, method='L-BFGS-B',   options={'maxiter': n_iter})
        train_step.minimize(self.sess, feed_dict={X: training_data, Y: output_data})

