import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

import scipy.io
import pandas as pd
from sklearn.model_selection import train_test_split


class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

nFeatures = 59*4 #+1 to account for droplet diameter size
latent_dim = 16
initNNodes = 512

alpha = 0.3
encoder_inputs = keras.Input(shape=nFeatures)
nNodes = initNNodes

enc = keras.layers.Dense(initNNodes)(encoder_inputs)
enc = keras.layers.LeakyReLU(alpha)(enc)
enc = keras.layers.Dropout(0.1)(enc)
enc = keras.layers.BatchNormalization()(enc)
enc = keras.layers.LeakyReLU(alpha)(enc)
enc = keras.layers.Dropout(0.1)(enc)
enc = keras.layers.BatchNormalization()(enc)
z_mean = layers.Dense(latent_dim, name="z_mean")(enc)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(enc)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
n = latent_dim

dec = keras.layers.Dense(initNNodes)(latent_inputs)
dec = keras.layers.LeakyReLU(alpha)(dec)
dec = keras.layers.Dropout(0.1)(dec)
dec = keras.layers.BatchNormalization()(dec)
dec = keras.layers.LeakyReLU(alpha)(dec)
dec = keras.layers.Dropout(0.1)(dec)
dec = keras.layers.BatchNormalization()(dec)
decoder_outputs = keras.layers.Dense(nFeatures, activation='tanh')(dec)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            #Mean squared Error as loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = encoder(data)
        reconstruction = decoder(z)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(data, reconstruction)
        )
        reconstruction_loss *= 28 * 28
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

input_filename = '/Users/cequilod/WaveSuite_VTK/sateNo4_1_pcs_uvwnut_data_40_to_99.npy'

trainingData = np.load(input_filename)

tf.random.set_seed(42)

def scaler(x, xmin, xmax, min, max):
    scale = (max - min) / (xmax - xmin)
    xScaled = scale * x + min - xmin * scale
    return xScaled
min_ls = np.min(trainingData)
print(min_ls)
max_ls = np.max(trainingData)
print(max_ls)
min = -1
max = 1

X_train, X_test, y_train, y_test = train_test_split(trainingData, trainingData, test_size=0.2, shuffle=True,
                                                    random_state=42)

data = scaler(X_train, min_ls, max_ls, min, max)
val_data = scaler(X_test, min_ls, max_ls, -1, 1)

print(data.shape)
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Nadam())
vae.fit(data, epochs=50, batch_size=8)#, validation_data=(X_test, y_test))

encoder.save('/Users/cequilod/WaveSuite_VTK/VAE_encoder')
decoder.save('/Users/cequilod/WaveSuite_VTK/VAE_decoder')

#####End of Training

def inverseScaler(xscaled, xmin, xmax, min, max):
    scale = (max - min) / (xmax - xmin)
    xInv = (xscaled/scale) - (min/scale) + xmin
    return xInv

def scaler(x, xmin, xmax, min, max):
    scale = (max - min) / (xmax - xmin)
    xScaled = scale * x + min - xmin * scale
    return xScaled

directory_data = '/Users/cequilod/WaveSuite_VTK/'
nameSim = 'sateNo4_1'
observationPeriod = 'data_40_to_99'

indices_filename = 'indices_test.npy'
indices_test = np.load(directory_data + indices_filename)

U_data_real = np.load(directory_data + nameSim + '_' + 'Uall' + '_' + observationPeriod + '.npy')
nut_data_real = np.load(directory_data + nameSim + '_' + 'nut' + '_' + observationPeriod + '.npy')

U_data_real = U_data_real[indices_test, :]
nut_data_real = nut_data_real[indices_test, :]

field_name = 'uvwnut'
print(field_name)
observationPeriod = 'data_40_to_99'
modelPcs = np.load(directory_data + nameSim + '_pcs_' + field_name + '_' + observationPeriod + '.npy')

generator_enc = load_model(directory_data +
                                     'VAE_encoder')
generator_dec = load_model(directory_data +
                                     'VAE_decoder')

xmin = np.min(modelPcs)
xmax = np.max(modelPcs)
import time
start = time.time()
input = scaler(modelPcs[indices_test, :], xmin, xmax, -1, +1)
#input = scaler(modelPcs, xmin, xmax, -1, +1)

pcae = generator_dec.predict(np.array(generator_enc.predict(input))[0, :])
ps_pcae = inverseScaler(pcae, xmin, xmax, -1, 1)

variableNames = ['u', 'v', 'w', 'nut']
Uallnutsep = []
j = 1
for varName in variableNames:
    modelEofs = np.load(directory_data + nameSim + '_eofs_' + varName + '_' + observationPeriod + '.npy')
    stdmodel = np.load(directory_data + nameSim + '_std_' + varName + '_' + observationPeriod + '.npy')
    meanmodel = np.load(directory_data + nameSim + '_mean_' + varName + '_' + observationPeriod + '.npy')

    Uallnutsep.append(np.squeeze(np.matmul(ps_pcae[:, 59*(j-1):59*j], modelEofs[:, :]) * stdmodel + meanmodel))
    j = j+1

u_sep2_VAE = Uallnutsep[0]
v_sep2_VAE = Uallnutsep[1]
w_sep2_VAE = Uallnutsep[2]
nut_sep2_VAE = Uallnutsep[3]

### Plots

import matplotlib.pyplot as plt
import seaborn as sns
plt.close('all')
fig = plt.figure(figsize=(20, 10))

plt.subplot(2,2,1)
sns.kdeplot(np.mean(U_data_real[:, :851101], 0))
#sns.kdeplot(np.mean(Uallsep_gen[:, :851101], 0), alpha=0.6)
#sns.kdeplot(np.mean(u_sep2_gen, 0), alpha=0.6)
#sns.kdeplot(np.mean(U_all_gen[:, :851101], 0), alpha=0.6)
sns.kdeplot(np.mean(u_sep2_VAE, 0), alpha=0.6)
plt.xlim(-2,2)
plt.title('u')
plt.legend(['Ground truth', 'Unut_sep (PC-AAE)', 'uvwnut (PC-AAE)', 'Unut (PC-AAE)', 'VAE'])

plt.subplot(2,2,2)
sns.kdeplot(np.mean(U_data_real[:, 851101:851101*2], 0))
#sns.kdeplot(np.mean(Uallsep_gen[:, 851101:851101*2], 0), alpha=0.6)
#sns.kdeplot(np.mean(v_sep2_gen, 0), alpha=0.6)
#sns.kdeplot(np.mean(U_all_gen[:, 851101:851101*2], 0), alpha=0.6)
sns.kdeplot(np.mean(v_sep2_VAE, 0), alpha=0.6)
plt.xlim(-2,2)
plt.title('v')

plt.subplot(2,2,3)
sns.kdeplot(np.mean(U_data_real[:, 851101*2:851101*3], 0), alpha=0.6)
#sns.kdeplot(np.mean(Uallsep_gen[:, 851101*2:851101*3], 0), alpha=0.6)
#sns.kdeplot(np.mean(w_sep2_gen, 0), alpha=0.6)
#sns.kdeplot(np.mean(U_all_gen[:, 851101*2:851101*3], 0), alpha=0.6)
sns.kdeplot(np.mean(w_sep2_VAE, 0), alpha=0.6)
plt.title('w')
plt.xlim(-2,2)

plt.subplot(2,2,4)
sns.kdeplot(np.mean(nut_data_real, 0), alpha=0.6)
#sns.kdeplot(np.mean(nutsep_gen, 0), alpha = 0.6)
#sns.kdeplot(np.mean(nut_sep2_gen, 0), alpha = 0.6)
#sns.kdeplot(np.mean(nut_all_gen, 0), alpha = 0.6)
sns.kdeplot(np.mean(nut_sep2_VAE, 0), alpha = 0.6)
plt.xlim(-1,2)
plt.title('Dynamic viscosity')
plt.tight_layout()