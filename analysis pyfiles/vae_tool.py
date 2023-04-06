#Genrate component function

from keras import backend as K
import numpy as np
from keras import losses
import pandas as pd
import matplotlib.pyplot as plt

#network parameter
batch_size = 48

def component_plot(decoder_model, encoded_array, latent_dim, std):
    """
    generate mean value data std = 0, 
    generate mean+std value data std = 1, 
    generate mean-std value data std = -1
    """
    copied_array = encoded_array.copy()
    copied_array[:, latent_dim] = copied_array[:, latent_dim] + std * np.std(copied_array, axis=0)[latent_dim]
    decoded_all = decoder_model.predict(copied_array, batch_size = batch_size)
    return np.mean(decoded_all, axis=0)

def loss_fn(x, x_decoded_mean):
    xent_loss = losses.binary_crossentropy(x, x_decoded_mean)
    kl_loss = -0.5 * K.mean(1+z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

def vae_figure(input_dataset, latent_dim, columns_index, encoder_model, decoder_model):
    figure = plt.figure(figsize = (16, 10))
    encoded_data = encoder_model.predict(input_dataset, batch_size = 48)
    for i in range(latent_dim):
        axe = plt.subplot(2, 4, i+1)
        axe.plot(columns_index, component_plot(decoder_model, encoded_data, i, 1), label = 'mean+std')
        axe.plot(columns_index, component_plot(decoder_model, encoded_data, i, 0), label = 'mean')
        axe.plot(columns_index, component_plot(decoder_model, encoded_data, i, -1), label = 'mean-std')
        axe.set_xlabel('Wavelength')
        axe.set_title('Latent Dimension ' + str(i+1))
        plt.legend(loc = 'upper right')
    plt.tight_layout()
    return figure

