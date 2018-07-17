import tensorflow as tf

class VAE:

    def __init__(self,encoder,decoder,latents):
        '''
        Description: This initializer takes encoder and decoder functions, along with a latent variable description and constructs the VAE loss functions and evaluation functions.

        args: 
         encoder - a function that takes an input and outputs 

        '''

        self.encoder = encoder
