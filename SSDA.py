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

