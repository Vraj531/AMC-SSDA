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
