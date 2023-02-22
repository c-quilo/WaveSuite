import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as tf

directory_data = '/Users/cequilod/WaveSuite_VTK/'
#directory_data = '/Users/cequilod/VTK/'

def inverseScaler(xscaled, xmin, xmax, min, max):
    scale = (max - min) / (xmax - xmin)
    xInv = (xscaled/scale) - (min/scale) + xmin
    return xInv

#Load generator
epoch = 7000
ls = 16
nSamples = 100
#nameSim = 'sateNo4_1'
nameSim = 'sateNo4_1'
field_name = 'Uallnutsep'
observationPeriod = 'data_40_to_99'
#observationPeriod = 'data_1_to_131'
experiment = field_name
modelPcs = np.load(directory_data + nameSim + '_pcs_' + field_name + '_' + observationPeriod + '.npy')

#Load nut
U_data_real = np.load(directory_data + nameSim + '_' + 'Uall' + '_' + observationPeriod + '.npy')
nut_data_real = np.load(directory_data + nameSim + '_' + 'nut' + '_' + observationPeriod + '.npy')

generator_dec = tf.models.load_model(directory_data +
                                     'AAE_MV_generator_decoder_Full_1WGAN_' +
                                     field_name +
                                     '_' +
                                     str(ls) +
                                     '_' +
                                     str(epoch))
xmin = np.min(modelPcs)
xmax = np.max(modelPcs)
import time
start = time.time()
noise = np.random.normal(0, 1, size=(nSamples, ls))
pcae = generator_dec.predict(noise)
ps_pcae = inverseScaler(pcae, xmin, xmax, -1, 1)

#back to PS
variableNames = ['Uall', 'nut']
Uallnutsep = []
j = 1
for varName in variableNames:
    modelEofs = np.load(directory_data + nameSim + '_eofs_' + varName + '_' + observationPeriod + '.npy')
    stdmodel = np.load(directory_data + nameSim + '_std_' + varName + '_' + observationPeriod + '.npy')
    meanmodel = np.load(directory_data + nameSim + '_mean_' + varName + '_' + observationPeriod + '.npy')

    Uallnutsep.append(np.squeeze(np.matmul(ps_pcae[:, 59*(j-1):59*j], modelEofs[:, :]) * stdmodel + meanmodel))
    j = j+1
Uallsep_gen = Uallnutsep[0]
nutsep_gen = Uallnutsep[1]

print(time.time() - start)

#Reshape U

#y_pred = np.load(directory_data + 'y_pred.npy')
#data = np.load(directory_data + 'sateNo4_1_nut_data_40_to_99.npy')
row = 3
fig = plt.figure(figsize=(20,10))
plt.xticks([])
plt.yticks([])
cmap = 'jet'

p = pv.Plotter(shape=(row, row), off_screen=True)
count = 0
for i in range(row):
    for j in range(row):

        p.subplot(i, j)
        mesh = pv.read(directory_data + 'sateNo4_1_' + str(40) + '.vtu')
        #mesh = pv.read(directory_data + 'regularWave_' + str(1) + '.vtu')

        #mesh['vtkCompositeIndex'] = np.reshape(U_gen[count, :], (nutsep_gen.shape[1], 3), order='F')
        mesh['nut'] = pv.pyvista_ndarray(nutsep_gen[count, :])
        #mesh.set_active_scalars('vtkCompositeIndex', 'point')
        mesh.set_active_scalars('nut', 'point')
        single_slice = mesh.slice(normal=[0, 1, 0])
        p.add_mesh(single_slice, cmap=cmap, clim=[0, 0.4])
        p.camera_position = 'xz'
        p.camera.zoom(1)
        p.set_background('white')
        print(count)
        count = count + 1
    #p.subplot(0, 1)
    #mesh_real = pv.read(directory_data + 'sateNo_4_1_modified_' + str(i + 40) + '.vtu')
    #mesh_real.set_active_scalars('vtkCompositeIndex')
    #single_slice = mesh_real.slice(normal=[1, 0, 0])
    #p.add_mesh(single_slice, cmap=cmap, clim=[0, 1.5])
    #p.camera_position = 'yz'
    #p.camera.zoom(1.7)

image = p.screenshot(None, return_img=True)
plt.imshow(image)
plt.tight_layout()
plt.savefig(directory_data + 'nut_prediction_xz_' + experiment + '_' + str(count))

#t-SNE

import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

U_mag_real = np.sqrt(U_data_real[:, :851101]**2 + U_data_real[:, 851101:851101*2]**2 + U_data_real[:,  851101*2:851101*3]**2)
U_mag_Uallnutsep = np.sqrt(Uallsep_gen[:, :851101]**2 + Uallsep_gen[:, 851101:851101*2]**2 + Uallsep_gen[:, 851101*2:851101*3]**2)

df1 = pd.DataFrame(U_mag_real[:, :])
df1['Group'] = 'U_GT'
#df2 = pd.DataFrame(Unut_data[:, 851101*3:851101*4])
df2 = pd.DataFrame(nut_data_real[:, :])
df2['Group'] = 'nut_GT'
df1 = df1.append(df2, ignore_index=True)

df3 = pd.DataFrame(U_mag_Uallnutsep[:, :])
df3['Group'] = 'U_PC-AAE'
df1 = df1.append(df3, ignore_index=True)

df4 = pd.DataFrame(nutsep_gen[:, :])
df4['Group'] = 'nut_PC-AAE'
df1 = df1.append(df4, ignore_index=True)

Group = df1["Group"]
df1 = df1.drop(labels = ["Group"],axis = 1)

Unut_gen_embedded = TSNE(n_components=2).fit_transform(df1)
tsne2done = Unut_gen_embedded[:, 0]
tsne2dtwo = Unut_gen_embedded[:, 1]
g = sns.scatterplot(
    x=tsne2done, y=tsne2dtwo,
    hue=Group,
    style=Group,
    data=df1,
    alpha=0.8,
    s=200,
    markers={'U_GT': 'o', 'U_PC-AAE': '^', 'nut_GT': 'o', 'nut_PC-AAE': '^'}
)
for lh in g._legend.legendHandles:
    lh.set_alpha(1)
    lh._sizes = [50]

##################################################
#Another Experiment u,v,w,nut

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as tf

directory_data = '/Users/cequilod/WaveSuite_VTK/'
#directory_data = '/Users/cequilod/VTK/'

def inverseScaler(xscaled, xmin, xmax, min, max):
    scale = (max - min) / (xmax - xmin)
    xInv = (xscaled/scale) - (min/scale) + xmin
    return xInv

#Load generator
epoch = 7000
ls = 16
nSamples = 100
#nameSim = 'sateNo4_1'
nameSim = 'sateNo4_1'
field_name = 'uvwnut'
observationPeriod = 'data_40_to_99'
#observationPeriod = 'data_1_to_131'
experiment = field_name
modelPcs = np.load(directory_data + nameSim + '_pcs_' + field_name + '_' + observationPeriod + '.npy')

#Load nut
U_data_real = np.load(directory_data + nameSim + '_' + 'Uall' + '_' + observationPeriod + '.npy')
nut_data_real = np.load(directory_data + nameSim + '_' + 'nut' + '_' + observationPeriod + '.npy')

generator_dec = tf.models.load_model(directory_data +
                                     'AAE_MV_generator_decoder_Full_1WGAN_' +
                                     field_name +
                                     '_' +
                                     str(ls) +
                                     '_' +
                                     str(epoch))
xmin = np.min(modelPcs)
xmax = np.max(modelPcs)
import time
start = time.time()
noise = np.random.normal(0, 1, size=(nSamples, ls))
pcae = generator_dec.predict(noise)
ps_pcae = inverseScaler(pcae, xmin, xmax, -1, 1)

#back to PS
variableNames = ['u', 'v', 'w', 'nut']
Uallnutsep = []
j = 1
for varName in variableNames:
    modelEofs = np.load(directory_data + nameSim + '_eofs_' + varName + '_' + observationPeriod + '.npy')
    stdmodel = np.load(directory_data + nameSim + '_std_' + varName + '_' + observationPeriod + '.npy')
    meanmodel = np.load(directory_data + nameSim + '_mean_' + varName + '_' + observationPeriod + '.npy')

    Uallnutsep.append(np.squeeze(np.matmul(ps_pcae[:, 59*(j-1):59*j], modelEofs[:, :]) * stdmodel + meanmodel))
    j = j+1

u_sep2_gen = Uallnutsep[0]
v_sep2_gen = Uallnutsep[1]
w_sep2_gen = Uallnutsep[2]
nut_sep2_gen = Uallnutsep[3]

print(time.time() - start)

#Reshape U

#y_pred = np.load(directory_data + 'y_pred.npy')
#data = np.load(directory_data + 'sateNo4_1_nut_data_40_to_99.npy')
row = 10
fig = plt.figure(figsize=(20,10))
plt.xticks([])
plt.yticks([])
cmap = 'jet'

p = pv.Plotter(shape=(row, row), off_screen=True)
count = 0
for i in range(row):
    for j in range(row):

        p.subplot(i, j)
        mesh = pv.read(directory_data + 'sateNo4_1_' + str(40) + '.vtu')
        #mesh = pv.read(directory_data + 'regularWave_' + str(1) + '.vtu')

        #mesh['vtkCompositeIndex'] = np.reshape(U_gen[count, :], (nutsep_gen.shape[1], 3), order='F')
        mesh['nut'] = pv.pyvista_ndarray(nut_sep2_gen[count, :])
        #mesh.set_active_scalars('vtkCompositeIndex', 'point')
        mesh.set_active_scalars('nut', 'point')
        single_slice = mesh.slice(normal=[0, 1, 0])
        p.add_mesh(single_slice, cmap=cmap)#, clim=[0, 0.4])
        p.camera_position = 'xz'
        p.camera.zoom(1)
        p.set_background('white')
        print(count)
        count = count + 1
    #p.subplot(0, 1)
    #mesh_real = pv.read(directory_data + 'sateNo_4_1_modified_' + str(i + 40) + '.vtu')
    #mesh_real.set_active_scalars('vtkCompositeIndex')
    #single_slice = mesh_real.slice(normal=[1, 0, 0])
    #p.add_mesh(single_slice, cmap=cmap, clim=[0, 1.5])
    #p.camera_position = 'yz'
    #p.camera.zoom(1.7)

image = p.screenshot(None, return_img=True)
plt.imshow(image)
plt.tight_layout()
plt.savefig(directory_data + 'nut_prediction_xz_' + experiment + '_' + str(count))

U_mag_real = np.sqrt(U_data_real[:, :851101]**2 + U_data_real[:, 851101:851101*2]**2 + U_data_real[:, 851101*2:851101*3]**2)
U_mag_uvwnut = np.sqrt(u_sep2_gen**2 + v_sep2_gen**2 + w_sep2_gen**2)

df1 = pd.DataFrame(U_mag_real[:, :])
df1['Group'] = 'U_GT'
#df2 = pd.DataFrame(Unut_data[:, 851101*3:851101*4])
df2 = pd.DataFrame(nut_data_real[:, :])
df2['Group'] = 'nut_GT'
df1 = df1.append(df2, ignore_index=True)

df3 = pd.DataFrame(U_mag_uvwnut[:, :])
df3['Group'] = 'U_PC-AAE'
df1 = df1.append(df3, ignore_index=True)

df4 = pd.DataFrame(nut_sep2_gen[:, :])
df4['Group'] = 'nut_PC-AAE'
df1 = df1.append(df4, ignore_index=True)

Group = df1["Group"]
df1 = df1.drop(labels = ["Group"],axis = 1)

Unut_gen_embedded = TSNE(n_components=2).fit_transform(df1)
tsne2done = Unut_gen_embedded[:, 0]
tsne2dtwo = Unut_gen_embedded[:, 1]
g = sns.scatterplot(
    x=tsne2done, y=tsne2dtwo,
    hue=Group,
    style=Group,
    data=df1,
    alpha=0.8,
    s=200,
    markers={'U_GT': 'o', 'U_PC-AAE': '^', 'nut_GT': 'o', 'nut_PC-AAE': '^'}
)
for lh in g._legend.legendHandles:
    lh.set_alpha(1)
    lh._sizes = [50]

##################################################
#Another Experiment Unut

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as tf

directory_data = '/Users/cequilod/WaveSuite_VTK/'
#directory_data = '/Users/cequilod/VTK/'

def inverseScaler(xscaled, xmin, xmax, min, max):
    scale = (max - min) / (xmax - xmin)
    xInv = (xscaled/scale) - (min/scale) + xmin
    return xInv

#Load generator
epoch = 7000
ls = 16
nSamples = 100
#nameSim = 'sateNo4_1'
nameSim = 'sateNo4_1'
field_name = 'Unut'
observationPeriod = 'data_40_to_99'
#observationPeriod = 'data_1_to_131'
experiment = field_name
modelPcs = np.load(directory_data + nameSim + '_pcs_' + field_name + '_' + observationPeriod + '.npy')

#Load nut
U_data_real = np.load(directory_data + nameSim + '_' + 'Uall' + '_' + observationPeriod + '.npy')
nut_data_real = np.load(directory_data + nameSim + '_' + 'nut' + '_' + observationPeriod + '.npy')

generator_dec = tf.models.load_model(directory_data +
                                     'AAE_MV_generator_decoder_Full_1WGAN_' +
                                     field_name +
                                     '_' +
                                     str(ls) +
                                     '_' +
                                     str(epoch))
xmin = np.min(modelPcs)
xmax = np.max(modelPcs)
import time
start = time.time()
noise = np.random.normal(0, 1, size=(nSamples, ls))
pcae = generator_dec.predict(noise)
ps_pcae = inverseScaler(pcae, xmin, xmax, -1, 1)

#back to PS
variableNames = ['Unut']
Uallnutsep = []
j = 1
for varName in variableNames:
    modelEofs = np.load(directory_data + nameSim + '_eofs_' + varName + '_' + observationPeriod + '.npy')
    stdmodel = np.load(directory_data + nameSim + '_std_' + varName + '_' + observationPeriod + '.npy')
    meanmodel = np.load(directory_data + nameSim + '_mean_' + varName + '_' + observationPeriod + '.npy')

    Uallnutsep.append(np.squeeze(np.matmul(ps_pcae[:, 59*(j-1):59*j], modelEofs[:, :]) * stdmodel + meanmodel))
    j = j+1

U_all_gen = Uallnutsep[0][:, :851101*3]
nut_all_gen = Uallnutsep[0][:, -851101::]

print(time.time() - start)

#Reshape U

#y_pred = np.load(directory_data + 'y_pred.npy')
#data = np.load(directory_data + 'sateNo4_1_nut_data_40_to_99.npy')
row = 5
fig = plt.figure(figsize=(20,10))
plt.xticks([])
plt.yticks([])
cmap = 'jet'

p = pv.Plotter(shape=(row, row), off_screen=True)
count = 0
for i in range(row):
    for j in range(row):

        p.subplot(i, j)
        mesh = pv.read(directory_data + 'sateNo4_1_' + str(40) + '.vtu')
        #mesh = pv.read(directory_data + 'regularWave_' + str(1) + '.vtu')

        #mesh['vtkCompositeIndex'] = np.reshape(U_gen[count, :], (nutsep_gen.shape[1], 3), order='F')
        mesh['nut'] = pv.pyvista_ndarray(nut_all_gen[count, :])
        #mesh.set_active_scalars('vtkCompositeIndex', 'point')
        mesh.set_active_scalars('nut', 'point')
        single_slice = mesh.slice(normal=[0, 1, 0])
        p.add_mesh(single_slice, cmap=cmap)#, clim=[0, 0.4])
        p.camera_position = 'xz'
        p.camera.zoom(1)
        p.set_background('white')
        print(count)
        count = count + 1
    #p.subplot(0, 1)
    #mesh_real = pv.read(directory_data + 'sateNo_4_1_modified_' + str(i + 40) + '.vtu')
    #mesh_real.set_active_scalars('vtkCompositeIndex')
    #single_slice = mesh_real.slice(normal=[1, 0, 0])
    #p.add_mesh(single_slice, cmap=cmap, clim=[0, 1.5])
    #p.camera_position = 'yz'
    #p.camera.zoom(1.7)

image = p.screenshot(None, return_img=True)
plt.imshow(image)
plt.tight_layout()
plt.savefig(directory_data + 'nut_prediction_xz_' + experiment + '_' + str(count))

U_mag_real = np.sqrt(U_data_real[:, :851101]**2 + U_data_real[:, 851101:851101*2]**2 + U_data_real[:,  851101*2:851101*3]**2)
U_mag_Unut = np.sqrt(U_all_gen[:, :851101]**2 + U_all_gen[:, 851101:851101*2]**2 + U_all_gen[:,  851101*2:851101*3]**2)

df1 = pd.DataFrame(U_mag_real[:, :])
df1['Group'] = 'U_GT'
#df2 = pd.DataFrame(Unut_data[:, 851101*3:851101*4])
df2 = pd.DataFrame(nut_data_real[:, :])
df2['Group'] = 'nut_GT'
df1 = df1.append(df2, ignore_index=True)

df3 = pd.DataFrame(U_mag_sep[:, :])
df3['Group'] = 'U_PC-AAE'
df1 = df1.append(df3, ignore_index=True)

df4 = pd.DataFrame(nut_all_gen[:, :])
df4['Group'] = 'nut_PC-AAE'
df1 = df1.append(df4, ignore_index=True)

Group = df1["Group"]
df1 = df1.drop(labels = ["Group"],axis = 1)

Unut_gen_embedded = TSNE(n_components=2).fit_transform(df1)
tsne2done = Unut_gen_embedded[:, 0]
tsne2dtwo = Unut_gen_embedded[:, 1]
g = sns.scatterplot(
    x=tsne2done, y=tsne2dtwo,
    hue=Group,
    style=Group,
    data=df1,
    alpha=0.8,
    s=200,
    markers={'U_GT': 'o', 'U_PC-AAE': '^', 'nut_GT': 'o', 'nut_PC-AAE': '^'}
)
for lh in g._legend.legendHandles:
    lh.set_alpha(1)
    lh._sizes = [50]

##################################################
#Another Experiment Unut VAE

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as tf

directory_data = '/Users/cequilod/WaveSuite_VTK/'
#directory_data = '/Users/cequilod/VTK/'

def inverseScaler(xscaled, xmin, xmax, min, max):
    scale = (max - min) / (xmax - xmin)
    xInv = (xscaled/scale) - (min/scale) + xmin
    return xInv

#Load generator
epoch = 7000
ls = 16
nSamples = 100
#nameSim = 'sateNo4_1'
nameSim = 'sateNo4_1'
field_name = 'uvwnut'
observationPeriod = 'data_40_to_99'

print(time.time() - start)

modelPcs = np.load(directory_data + nameSim + '_pcs_' + field_name + '_' + observationPeriod + '.npy')

#Load nut
U_data_real = np.load(directory_data + nameSim + '_' + 'Uall' + '_' + observationPeriod + '.npy')
nut_data_real = np.load(directory_data + nameSim + '_' + 'nut' + '_' + observationPeriod + '.npy')

generator_dec = tf.models.load_model(directory_data +
                                     'VAE_decoder')
xmin = np.min(modelPcs)
xmax = np.max(modelPcs)
import time
start = time.time()
noise = np.random.normal(0, 1, size=(nSamples, ls))
pcae = generator_dec.predict(noise)
ps_pcae = inverseScaler(pcae, xmin, xmax, -1, 1)

#back to PS
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
print(time.time()-start)


row = 5
fig = plt.figure(figsize=(20,10))
plt.xticks([])
plt.yticks([])
cmap = 'jet'

p = pv.Plotter(shape=(row, row), off_screen=True)
count = 0
for i in range(row):
    for j in range(row):

        p.subplot(i, j)
        mesh = pv.read(directory_data + 'sateNo4_1_' + str(40) + '.vtu')
        #mesh = pv.read(directory_data + 'regularWave_' + str(1) + '.vtu')

        #mesh['vtkCompositeIndex'] = np.reshape(U_gen[count, :], (nutsep_gen.shape[1], 3), order='F')
        mesh['nut'] = pv.pyvista_ndarray(nut_sep2_VAE[count, :])
        #mesh.set_active_scalars('vtkCompositeIndex', 'point')
        mesh.set_active_scalars('nut', 'point')
        single_slice = mesh.slice(normal=[0, 1, 0])
        p.add_mesh(single_slice, cmap=cmap)#, clim=[0, 0.4])
        p.camera_position = 'xz'
        p.camera.zoom(1)
        p.set_background('white')
        print(count)
        count = count + 1
    #p.subplot(0, 1)
    #mesh_real = pv.read(directory_data + 'sateNo_4_1_modified_' + str(i + 40) + '.vtu')
    #mesh_real.set_active_scalars('vtkCompositeIndex')
    #single_slice = mesh_real.slice(normal=[1, 0, 0])
    #p.add_mesh(single_slice, cmap=cmap, clim=[0, 1.5])
    #p.camera_position = 'yz'
    #p.camera.zoom(1.7)

image = p.screenshot(None, return_img=True)
plt.imshow(image)
plt.tight_layout()
plt.savefig(directory_data + 'nut_prediction_xz_' + experiment + '_VAE_' + str(count))

U_mag_real = np.sqrt(U_data_real[:, :851101]**2 + U_data_real[:, 851101:851101*2]**2 + U_data_real[:,  851101*2:851101*3]**2)
U_mag_VAE = np.sqrt(u_sep2_VAE**2 + v_sep2_VAE**2 + w_sep2_VAE**2)

df1 = pd.DataFrame(U_mag_real[:, :])
df1['Group'] = 'U_GT'
#df2 = pd.DataFrame(Unut_data[:, 851101*3:851101*4])
df2 = pd.DataFrame(nut_data_real[:, :])
df2['Group'] = 'nut_GT'
df1 = df1.append(df2, ignore_index=True)

df3 = pd.DataFrame(U_mag_VAE[:, :])
df3['Group'] = 'U_VAE'
df1 = df1.append(df3, ignore_index=True)

df4 = pd.DataFrame(nut_sep2_VAE[:, :])
df4['Group'] = 'nut_VAE'
df1 = df1.append(df4, ignore_index=True)

Group = df1["Group"]
df1 = df1.drop(labels = ["Group"],axis = 1)

Unut_gen_embedded = TSNE(n_components=2).fit_transform(df1)
tsne2done = Unut_gen_embedded[:, 0]
tsne2dtwo = Unut_gen_embedded[:, 1]
g = sns.scatterplot(
    x=tsne2done, y=tsne2dtwo,
    hue=Group,
    style=Group,
    data=df1,
    alpha=0.8,
    s=200,
    markers={'U_GT': 'o', 'U_VAE': '^', 'nut_GT': 'o', 'nut_VAE': '^'}
)
for lh in g._legend.legendHandles:
    lh.set_alpha(1)
    lh._sizes = [50]


#Explore the original dataset

pred_error = np.mean(U_mag_real, 0)
ci = 1 * np.std(U_mag_real, 0)  # / np.mean(error_temp, 0)
y = pred_error
plt.fill_between(range(pred_error.shape[0]), (y - ci), (y + ci), color='r', alpha=.3)

pred_error = np.mean(U_mag_Uallnutsep, 0)
ci = 1 * np.std(U_mag_Uallnutsep, 0)  # / np.mean(error_temp, 0)
y = pred_error
plt.fill_between(range(pred_error.shape[0]), (y - ci), (y + ci), color='b', alpha=.3)

pred_error = np.mean(U_mag_uvwnut, 0)
ci = 1 * np.std(U_mag_uvwnut, 0)  # / np.mean(error_temp, 0)
y = pred_error
plt.fill_between(range(pred_error.shape[0]), (y - ci), (y + ci), color='g', alpha=.3)

pred_error = np.mean(U_mag_Unut, 0)
ci = 1 * np.std(U_mag_Unut, 0)  # / np.mean(error_temp, 0)
y = pred_error
plt.fill_between(range(pred_error.shape[0]), (y - ci), (y + ci), color='r', alpha=.3)

pred_error = np.mean(U_mag_VAE, 0)
ci = 1 * np.std(U_mag_VAE, 0)  # / np.mean(error_temp, 0)
y = pred_error
plt.fill_between(range(pred_error.shape[0]), (y - ci), (y + ci), color='b', alpha=.3)


###Plot distributions
fig = plt.figure(figsize=(20, 10))

plt.subplot(2,2,1)
sns.kdeplot(np.mean(U_data_real[:, :851101], 0))
sns.kdeplot(np.mean(Uallsep_gen[:, :851101], 0), alpha=0.6)
sns.kdeplot(np.mean(u_sep2_gen, 0), alpha=0.6)
sns.kdeplot(np.mean(U_all_gen[:, :851101], 0), alpha=0.6)
sns.kdeplot(np.mean(u_sep2_VAE, 0), alpha=0.6)
plt.xlim(-2,2)
plt.title('u')
plt.legend(['Ground truth', 'Unut_sep (PC-AAE)', 'uvwnut (PC-AAE)', 'Unut (PC-AAE)', 'VAE'])

plt.subplot(2,2,2)
sns.kdeplot(np.mean(U_data_real[:, 851101:851101*2], 0))
sns.kdeplot(np.mean(Uallsep_gen[:, 851101:851101*2], 0), alpha=0.6)
sns.kdeplot(np.mean(v_sep2_gen, 0), alpha=0.6)
sns.kdeplot(np.mean(U_all_gen[:, 851101:851101*2], 0), alpha=0.6)
sns.kdeplot(np.mean(v_sep2_VAE, 0), alpha=0.6)
plt.xlim(-2,2)
plt.title('v')

plt.subplot(2,2,3)
sns.kdeplot(np.mean(U_data_real[:, 851101*2:851101*3], 0), alpha=0.6)
sns.kdeplot(np.mean(Uallsep_gen[:, 851101*2:851101*3], 0), alpha=0.6)
sns.kdeplot(np.mean(w_sep2_gen, 0), alpha=0.6)
sns.kdeplot(np.mean(U_all_gen[:, 851101*2:851101*3], 0), alpha=0.6)
sns.kdeplot(np.mean(w_sep2_VAE, 0), alpha=0.6)
plt.title('w')
plt.xlim(-2,2)

plt.subplot(2,2,4)
sns.kdeplot(np.mean(nut_data_real, 0), alpha=0.6)
sns.kdeplot(np.mean(nutsep_gen, 0), alpha = 0.6)
sns.kdeplot(np.mean(nut_sep2_gen, 0), alpha = 0.6)
sns.kdeplot(np.mean(nut_all_gen, 0), alpha = 0.6)
sns.kdeplot(np.mean(nut_sep2_VAE, 0), alpha = 0.6)
plt.xlim(-1,2)
plt.title('Dynamic viscosity')
plt.tight_layout()





np.sort()
np.sort(np.mean(U_mag_sep, 0))
#Plot u,v,w vs nut

fig = plt.figure(figsize=(20,10))
plt.subplot(1,3,1)

plt.scatter(np.mean(U_data_real[:, :851101], 1), np.mean(nut_data_real, 1), alpha=0.5)
plt.scatter(np.mean(Uallsep_gen[:, :851101], 1), np.mean(nutsep_gen, 1), alpha=0.5)
plt.scatter(np.mean(u_sep2_gen[:, :], 1), np.mean(nut_sep2_gen, 1), alpha=0.5)
plt.scatter(np.mean(U_all_gen[:, :851101], 1), np.mean(nut_all_gen, 1), alpha=0.5)
plt.scatter(np.mean(u_sep2_VAE[:, :], 1), np.mean(nut_sep2_VAE, 1), alpha=0.5)
plt.legend(['Ground truth', 'Unut_sep (PC-AAE)', 'uvwnut (PC-AAE)', 'Unut (PC-AAE)', 'VAE'])
plt.title('u')
plt.ylabel('Dynamic Viscosity  (N s/m2)')
plt.xlabel('Velocity (m s-1)')
plt.tight_layout()

plt.subplot(1,3,2)
plt.scatter(np.mean(U_data_real[:, 851101:851101*2], 1), np.mean(nut_data_real, 1), alpha=0.5)
plt.scatter(np.mean(Uallsep_gen[:, 851101:851101*2], 1), np.mean(nutsep_gen, 1), alpha=0.5)
plt.scatter(np.mean(v_sep2_gen[:, :], 1), np.mean(nut_sep2_gen, 1), alpha=0.5)
plt.scatter(np.mean(U_all_gen[:, 851101:851101*2], 1), np.mean(nut_all_gen, 1), alpha=0.5)
plt.scatter(np.mean(v_sep2_VAE[:, :], 1), np.mean(nut_sep2_VAE, 1), alpha=0.5)
plt.title('v')
plt.ylabel('Dynamic Viscosity  (N s/m2)')
plt.xlabel('Velocity (m s-1)')
plt.tight_layout()

plt.subplot(1,3,3)
plt.scatter(np.mean(U_data_real[:, 851101*2:851101*3], 1), np.mean(nut_data_real, 1), alpha=0.5)
plt.scatter(np.mean(Uallsep_gen[:, 851101*2:851101*3], 1), np.mean(nutsep_gen, 1), alpha=0.5)
plt.scatter(np.mean(w_sep2_gen[:, :], 1), np.mean(nut_sep2_gen, 1), alpha=0.5)
plt.scatter(np.mean(U_all_gen[:, 851101*2:851101*3], 1), np.mean(nut_all_gen, 1), alpha=0.5)
plt.scatter(np.mean(w_sep2_VAE[:, :], 1), np.mean(nut_sep2_VAE, 1), alpha=0.5)
plt.title('w')
plt.ylabel('Dynamic Viscosity  (N s/m2)')
plt.xlabel('Velocity (m s-1)')
plt.tight_layout()

##Scatter per point
fig = plt.figure(figsize=(20,10))
plt.subplot(1,3,1)

plt.scatter(np.mean(U_data_real[:, :851101], 0), np.mean(nut_data_real, 0), alpha=1)
#plt.plot(np.sort(np.mean(Uallsep_gen[:, :851101], 0)), np.sort(np.mean(nutsep_gen, 0)), alpha=0.5)
#plt.plot(np.sort(np.mean(u_sep2_gen[:, :], 0)), np.sort(np.mean(nut_sep2_gen, 0)), alpha=0.5)
plt.scatter(np.mean(U_all_gen[:, :851101], 0), np.mean(nut_all_gen, 0), alpha=0.1)
#plt.plot(np.sort(np.mean(u_sep2_VAE[:, :], 0)), np.sort(np.mean(nut_sep2_VAE, 0)), alpha=0.5)
plt.legend(['Ground truth', 'Unut_sep (PC-AAE)', 'uvwnut (PC-AAE)', 'Unut (PC-AAE)', 'VAE'])
plt.title('u')
plt.ylabel('Dynamic Viscosity  (N s/m2)')
plt.xlabel('Velocity (m s-1)')
plt.tight_layout()

plt.subplot(1,3,2)
plt.scatter(np.mean(U_data_real[:, 851101:851101*2], 0), np.mean(nut_data_real, 0), alpha=1)
#plt.plot(np.sort(np.mean(Uallsep_gen[:, 851101:851101*2], 0)), np.sort(np.mean(nutsep_gen, 0)), alpha=0.5)
#plt.plot(np.sort(np.mean(v_sep2_gen[:, :], 0)), np.sort(np.mean(nut_sep2_gen, 0)), alpha=0.5)
plt.scatter(np.mean(U_all_gen[:, 851101:851101*2], 0), np.mean(nut_all_gen, 0), alpha=0.1)
#plt.plot(np.sort(np.mean(v_sep2_VAE[:, :], 0)), np.sort(np.mean(nut_sep2_VAE, 0)), alpha=0.5)
plt.title('v')
plt.ylabel('Dynamic Viscosity  (N s/m2)')
plt.xlabel('Velocity (m s-1)')
plt.tight_layout()

plt.subplot(1,3,3)
plt.scatter(np.mean(U_data_real[:, 851101*2:851101*3], 0), np.mean(nut_data_real, 0), alpha=1)
#plt.plot(np.sort(np.mean(Uallsep_gen[:, 851101*2:851101*3], 0)), np.sort(np.mean(nutsep_gen, 0)), alpha=0.5)
#plt.plot(np.sort(np.mean(w_sep2_gen[:, :], 0)), np.sort(np.mean(nut_sep2_gen, 0)), alpha=0.5)
plt.scatter(np.mean(U_all_gen[:, 851101*2:851101*3], 0), np.mean(nut_all_gen, 0), alpha=0.1)
#plt.plot(np.sort(np.mean(w_sep2_VAE[:, :], 0)), np.sort(np.mean(nut_sep2_VAE, 0)), alpha=0.5)
plt.title('w')
plt.ylabel('Dynamic Viscosity  (N s/m2)')
plt.xlabel('Velocity (m s-1)')
plt.tight_layout()
