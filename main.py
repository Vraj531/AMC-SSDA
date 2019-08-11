#import libraries
import display_digits
import keras_AE
import PSNR
import FeedForward_sparse_AE
import SSDA
from cvxopt import matrix 
from cvxopt import solvers
import numpy 
from cvxopt import matrix 
from cvxopt import solvers 
import numpy as np
from neupy import algorithms
import os


pp='../input/mnist-train/'

from scipy.io import loadmat
images1 = loadmat(pp+'gaussianIMAGES.mat',variable_names='IMAGES',appendmat=True).get('IMAGES')
images = loadmat(pp+'IMAGES.mat',variable_names='IMAGES',appendmat=True).get('IMAGES')
x=images.astype(np.float32)
q = []
for _ in range(x.shape[2]):
    q.append(x[:,:,_].reshape(784))
q = np.array(q)

q1 = []
x1=images1.astype(np.float32)
for _ in range(x1.shape[2]):
    q1.append(x1[:,:,_].reshape(784))
q1 = np.array(q1)

 

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
    '''
      This is the implementation of the sparse autoencoder for https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf

    '''
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
        rho_hat=tf.reduce_mean(H,axis=0)   #Average hidden layer over all data points in X, Page 14 in https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
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


def visualizeW1(images, vis_patch_side, hid_patch_side, iter, file_name="trained_"):
    """ Visual all images in one pane"""

    figure, axes = matplotlib.pyplot.subplots(nrows=hid_patch_side, ncols=hid_patch_side)
    index = 0

    for axis in axes.flat:
        """ Add row of weights as an image to the plot """

        image = axis.imshow(images[index, :].reshape(vis_patch_side, vis_patch_side),
                            cmap=matplotlib.pyplot.cm.gray, interpolation='nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    """ Show the obtained plot """
    file=file_name+str(iter)+".png"
    matplotlib.pyplot.savefig(file)
    print("Written into "+ file)
    matplotlib.pyplot.close()


# Create SSDA
def create_ssda(x_train,x_test,noisy_data,noisy_data_test,n_inputs=784,n_hidden_outer=256,n_hidden_inner=256,start=0,lens=60000,learning_rate=0.01,n_iters_outer=50,n_iters_inner=50):
  
  #Converting to float32 for internal matrix multiplication coherence
  noisy_data = noisy_data.astype(np.float32)
  noisy_data_test = noisy_data_test.astype(np.float32)
  x_train = x_train.astype(np.float32)
  x_test = x_test.astype(np.float32)

  #Creating the AutoEncoder Objects
  outer_ae = FeedforwardSparseAutoEncoder(n_inputs,n_hidden_outer)
  inner_ae = FeedforwardSparseAutoEncoder(n_hidden_outer,n_hidden_inner)

  #Training the Outer Auto_Encoder with Noisy Data
  outer_ae.training(noisy_data[start:start+lens],x_train[start:start+lens],n_iter=n_iters_outer)
  latent1 = outer_ae.encode(noisy_data[start:start+lens]).eval(session=outer_ae.sess)

  #Coverting to float32 type
  latent1 = latent1.astype(np.float32)
  print('Outer Auto-Encoder Training Complete')
  #Training the Inner Auto_Encoder with Latent Vector from hidden layer of Outer Auto_Encoder
  inner_ae.training(latent1,latent1,n_iter=n_iters_inner)
  print('Inner Auto-Encoder Training Complete')
  return outer_ae,inner_ae

# Peak Signal-to-Noise Ratio
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
      return 100
    PIXEL_MAX = 1.
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# Display digital digits
def displayDigits(X,Y,start=0):
  n = 10  # of digits
  plt.figure(figsize=(20, 4))
  for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X[start+i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(Y[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()

# Create keras autoencoder
import keras.backend as K
def Kpsnr(y_true,y_pred):
    mse = K.mean(K.square(y_true-y_pred))
    return -4.343*K.log(mse)
# This is for user understanding part 
# This part is not used in this program.
def create_keras_ae(weights):
  input_image = Input(shape=(784,))
  outer_encoded = Dense(256,activation='sigmoid',activity_regularizer=regularizers.l2(10e-5))(input_image)
  inner_encoded = Dense(256,activation='sigmoid',activity_regularizer=regularizers.l2(10e-5))(outer_encoded)
  inner_decoded = Dense(256,activation='sigmoid')(inner_encoded)
  outer_decoded = Dense(784,activation='sigmoid')(inner_decoded)
 
  #Creating the complete SSDA
  keras_ae = keras.Model(input_image,outer_decoded)
  keras_ae.compile(optimizer='adam',loss='mse',metrics=[Kpsnr])
  keras_ae.summary()
    
  #Creating outer and inner encoder models for getting the hidden layer representations
  outer_encoder = keras.Model(input_image,outer_encoded)
  Latent_Input = Input(shape=(256,))
  In_Encoder_Layer = keras_ae.layers[-3]
  inner_encoder = keras.Model(Latent_Input,In_Encoder_Layer(Latent_Input))
  
 
  
  
  #Fetching W (weights) from weights
  W1 = weights.W1
  W2 = weights.W2
  W3 = weights.W3
  W4 = weights.W4
  
  #Fetching and Reshaping b (biases) from weights for setting it in the keras layers
  b1 = weights.b1.reshape(256,)
  b2 = weights.b2.reshape(256,)
  b3 = weights.b3.reshape(256,)
  b4 = weights.b4.reshape(784,)
  
  #Transferring weights from the Tensorflow SSDA model to the Keras model for FineTuning
  keras_ae.layers[1].set_weights([W1,b1])
  keras_ae.layers[2].set_weights([W2,b2])
  keras_ae.layers[3].set_weights([W3,b3])
  keras_ae.layers[4].set_weights([W4,b4])
  
 
  
  return keras_ae,outer_encoder,inner_encoder
  
path= '../input/results/results/'

# Load Keras


def load_keras(name):
    name = path+name
    # load json and create model
    json_file = open(name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name+'.h5')
    print("Loaded model from disk")
    return loaded_model



    
#Gaussian
g_keras_ae = load_keras('gaussian_keras_ae')
g_outer_encoder = load_keras('gaussian_outer_encoder')
g_inner_encoder = load_keras('gaussian_inner_encoder')
g_keras_ae.compile(optimizer='adam',loss='mse')
print('Gaussian Fine-Tuning Complete !')
print('----------------------------------------------------------------')

# Speckle
s_keras_ae = load_keras('speckle_keras_ae')
s_outer_encoder = load_keras('speckle_outer_encoder')
s_inner_encoder = load_keras('speckle_inner_encoder')
s_keras_ae.compile(optimizer='adam',loss='mse')
print('Speckle SSDA Fine-Tuning Complete !')
print('----------------------------------------------------------------')

# S&P
sp_keras_ae = load_keras('sp_keras_ae')
sp_outer_encoder = load_keras('sp_outer_encoder')
sp_inner_encoder = load_keras('sp_inner_encoder')
sp_keras_ae.compile(optimizer='adam',loss='mse')
print('S&P SSDA Fine-Tuning Complete !')
print('----------------------------------------------------------------')

# Border
bor_keras_ae = load_keras('bor_keras_ae')
bor_outer_encoder = load_keras('bor_outer_encoder')
bor_inner_encoder = load_keras('bor_inner_encoder')
bor_keras_ae.compile(optimizer='adam',loss='mse')
print('Border SSDA Fine-Tuning Complete !')
print('----------------------------------------------------------------')

# Block
blk_keras_ae = load_keras('blk_keras_ae')
blk_outer_encoder = load_keras('blk_outer_encoder')
blk_inner_encoder = load_keras('blk_inner_encoder')
blk_keras_ae.compile(optimizer='adam',loss='mse')
print('Block SSDA Fine-Tuning Complete !')
print('----------------------------------------------------------------')

# #Weight Prediction Module : Using same input image for all SSDAs and getting the Latent Vectors
def get_latent_stack(x):
#     temp = x.copy()
    # Clean
#     x = temp.astype(np.float32)
#     Latent_Clean = outer_encoder.predict(x)
#     y_pred_clean = keras_ae.predict(x)
#     # Comparing prediction with the original images
#     print('------------------------- Clean SSDA ------------------------')
#     displayDigits(x_train,y_pred_clean)
#     print(psnr(x_train,y_pred_clean))

    #Gaussian
#     x_train = temp.astype(np.float32)
    Latent_Gaussian = g_outer_encoder.predict(x)
#     y_pred_gaussian = g_keras_ae.predict(x)
    # Comparing prediction with the original images
    print('------------------------- Gaussian SSDA ------------------------')
#     displayDigits(x_train,y_pred_gaussian)
#     print(psnr(x_train,y_pred_gaussian))

    # Speckle
#     x_train = temp.astype(np.float32)
    Latent_Speckle = s_outer_encoder.predict(x)
#     y_pred_s = s_keras_ae.predict(x)
    # Comparing prediction with the original images
    print('------------------------- Speckle SSDA ------------------------')
#     displayDigits(x_train,y_pred_s)
#     print(psnr(x_train,y_pred_s))

    # S&P
#     x_train = temp.astype(np.float32)
    Latent_SP = sp_outer_encoder.predict(x)
#     y_pred_sp = sp_keras_ae.predict(x)
    # Comparing prediction with the original images
    print('------------------------- S&P SSDA ------------------------')
#     displayDigits(x_train,y_pred_sp)
#     print(psnr(x_train,y_pred_sp))

    # Border
#     x_train = temp.astype(np.float32)
    Latent_Border = bor_outer_encoder.predict(x)
#     y_pred_border = bor_keras_ae.predict(x)
    # Comparing prediction with the original images
    print('------------------------- Border SSDA ------------------------')
#     displayDigits(x_train,y_pred_border)
#     print(psnr(x_train,y_pred_border))

    # Block
#     x_train = temp.astype(np.float32)
    Latent_Block= blk_outer_encoder.predict(x)
#     y_pred_block = blk_keras_ae.predict(x)
    # Comparing prediction with the original images
    print('------------------------- Block SSDA ------------------------')
#     displayDigits(x_train,y_pred_block)
#     print(psnr(x_train,y_pred_block))
    
    Latent_train = np.concatenate((Latent_Gaussian,Latent_Speckle,Latent_SP,Latent_Border,Latent_Block),axis=1)
    return  Latent_train

ia=get_latent_stack(q)

# #Testing the SSDAs : 
temp = q1.astype(np.float32)


# Gaussian SSDA

y_pred_gaussian = g_keras_ae.predict(q1)
# Comparing prediction with the original images
print('------------------------- Gaussian SSDA ------------------------')
displayDigits(temp,y_pred_gaussian)
print(psnr(temp,y_pred_gaussian))

# Speckle SSDA
# x = speckle_n_data_test.astype(np.float32)
y_pred_s = s_keras_ae.predict(q1)
# Comparing prediction with the original images
print('------------------------- Speckle SSDA ------------------------')
displayDigits(temp,y_pred_s)
print(psnr(temp,y_pred_s))
 
# S&P SSDA
# x = sp_n_data_test.astype(np.float32)
y_pred_sp = sp_keras_ae.predict(q1)
# Comparing prediction with the original images
print('------------------------- S&P SSDA ------------------------')
displayDigits(temp,y_pred_sp)
print(psnr(temp,y_pred_sp))

# Border SSDA
# x = border_n_data_test.astype(np.float32)
y_pred_border = bor_keras_ae.predict(q1)
# Comparing prediction with the original images
print('------------------------- Border SSDA ------------------------')
displayDigits(temp,y_pred_border)
print(psnr(temp,y_pred_border))

# Block SSDA
# x = block_n_data_test.astype(np.float32)
y_pred_block = blk_keras_ae.predict(q1)
# Comparing prediction with the original images
print('------------------------- Block SSDA ------------------------')
displayDigits(temp,y_pred_block)
print(psnr(temp,y_pred_block))

t = q

y2=y_pred_gaussian.astype(np.float32)
y3=y_pred_s.astype(np.float32)
y4=y_pred_sp.astype(np.float32)
y5=y_pred_border.astype(np.float32)
y6=y_pred_block.astype(np.float32)

def to_sum(tmp):
    tt=tmp.flatten()
    tt=tt.sum(axis=0)

    return tt.item()

# Get weights
def get_weights(tmp,tmp2,tmp3,tmp4,tmp5,tmp6):
    t = tmp

    y2=tmp2
    y3=tmp3
    y4=tmp4
    y5=tmp5
    y6=tmp6

    t = to_sum(t)

    y2=to_sum(y2)
    y3=to_sum(y3)
    y4=to_sum(y4)
    y5=to_sum(y5)
    y6=to_sum(y6)

       
    P = matrix([
            [(y2*y2),0,0,0,0],
            [(2*y3*y2),(y3*y3),0,0,0],
            [(2*y4*y2),(2*y4*y3),(y4*y4),0,0],
            [(2*y5*y2),(2*y5*y3),(2*y5*y4),(y5*y5),0],
            [(2*y6*y2),(2*y6*y3),(2*y6*y4),(2*y6*y5),(y6*y6)]]) 
    q = matrix([(-2*t*y2),(-2*t*y3),(-2*t*y4),(-2*t*y5),(-2*t*y6)])
    G = matrix([[1.0,-1.0,-1.0,0.0,0.0,0.0,0.0],
            [1.0,-1.0,0.0,-1.0,0.0,0.0,0.0],
            [1.0,-1.0,0.0,0.0,-1.0,0.0,0.0],
            [1.0,-1.0,0.0,0.0,0.0,-1.0,0.0],
            [1.0,-1.0,0.0,0.0,0.0,0.0,-1.0],
           ]) 
    h = matrix([1.05,0.95,0.0,0.0,0.0,0.0,0.0])
    sol = solvers.qp(P,q,G,h)
    return sol

so=[]
for i in range(len(t)):
    so.append(get_weights(t[i],y2[i],y3[i],y4[i],y5[i],y6[i]))

xx=[]
v1=[]
for i in range(len(t)):
    xx=so[i]['x']
    xz=[xx[0],xx[1],xx[2],xx[3],xx[4]]
    xz = np.asarray(xz)
    xz=xz.reshape(5)
    v1.append(xz)
    

v1 = np.asarray(v1)

v=[]
for i in range(len(t)):
    v.append(((so[i]['x'][0]*y_pred_gaussian[i])+(so[i]['x'][1]*y_pred_s[i])+(so[i]['x'][2]*y_pred_sp[i])+(so[i]['x'][3]*y_pred_border[i])+(so[i]['x'][4]*y_pred_block[i]))/2)

#Here i showned a sample rbf to predict only one output of qp in similar way all 5 output of qp canbe predict from rbf
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from neupy import algorithms
# dataset = datasets.load_digits()
# x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3)
x_train=ia[0:20000,:]
y_train=v1[0:20000,0]

pnn = algorithms.PNN(std=10, verbose=False)
pnn.train(x_train, y_train)

# x_train=v[10000:20000,:]
# y_train=v1[10000:20000,0]

# pnn.train(x_train, y_train)
x_test=ia[20000:25000,:]
y_test=v1[20000:25000,0]
y_predicted = pnn.predict(x_test)
metrics.accuracy_score(y_test, y_predicted)



