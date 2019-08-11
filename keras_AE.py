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
  
