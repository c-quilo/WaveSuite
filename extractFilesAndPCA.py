import sys, os
import numpy as np
import pyvista as pv
import eofs
from eofs.standard import Eof
import vtk
import sys, os
sys.path.append('/Users/cequilod/')
import vtktools

class extractFieldsAndPCA():
    def __init__(self, directory_data, nameSimulation, field_name, velocityField, start, end, start_pca, end_pca, nsize):

        self.directory_data = directory_data
        self.nameSimulation = nameSimulation
        self.field_name = field_name
        self.start = start
        self.end = end
        self.nsize = nsize
        self.velocityField = velocityField

        # For PCA analysis specific time-steps
        self.start_pca = start_pca
        self.end_pca = end_pca

    def extractFields(self):
        tracer_data = np.zeros((self.end-self.start, self.nsize))
        if velocityField == True:
            field_data = np.zeros((self.end-self.start, self.nsize, 3))
        else:
            field_data = np.zeros((self.end - self.start, self.nsize))
        k = 0
        for i in np.arange(self.start, self.end):
                filename = self.directory_data + self.nameSimulation + str(i) + '.vtu'
                mesh = vtktools.vtu(filename)
                #field_data[k, :] = np.squeeze(mesh.GetField(self.field_name))
                field_data[k, :] = np.squeeze(mesh.GetLocations())
                print(k)
                k = k + 1

         np.save(self.directory_data + self.nameSimulation + self.field_name + '_' + 'data_' + str(self.start) + '_to_' + str(self.end), field_data)


    def PCA(self, field_name, velocityField, numberDimensions):

        field_name = field_name
        start_interv = self.start_pca
        end_interv = self.end_pca
        self.velocityField = velocityField
        self.numberDimensions = numberDimensions
        observationPeriod = 'data_' + str(start_interv) + '_to_' + str(end_interv)
        modelData = np.load(self.directory_data + self.nameSimulation + field_name + '_' + observationPeriod + '.npy')
        if velocityField == 0:
            modelData = modelData[:, :]

        # Velocity is a 3D vector and needs to be reshaped before the PCA
        elif velocityField == 1:
            print(modelData.shape)
            modelData = modelData[:, :, :self.numberDimensions]
            modelData = np.reshape(modelData, (modelData.shape[0], modelData.shape[1] * modelData.shape[2]), order='F')
        print(modelData.shape)
        # Standardise the data with mean 0
        meanData = np.nanmean(modelData, 0)
        stdData = np.nanstd(modelData)
        modelDataScaled = (modelData - meanData) / stdData

        #PCA solver
        solver = Eof(modelDataScaled)

        # Principal Components time-series
        pcs = solver.pcs()
        # Projection
        eof = solver.eofs()
        # Cumulative variance
        varianceCumulative = np.cumsum(solver.varianceFraction())

        np.save(self.directory_data + self.nameSimulation + 'pcs_' + field_name + '_' + observationPeriod,
                pcs)
        np.save(self.directory_data + self.nameSimulation + 'eofs_' + field_name + '_' + observationPeriod,
                eof)
        np.save(self.directory_data + self.nameSimulation + 'varCumulative_' + field_name + '_' + observationPeriod,
                varianceCumulative)
        np.save(self.directory_data + self.nameSimulation + 'mean_' + field_name + '_' + observationPeriod,
                meanData)
        np.save(self.directory_data + self.nameSimulation  + 'std_' + field_name + '_' + observationPeriod,
                stdData)

if __name__ == '__main__':
    directory_data = '/Users/cequilod/WaveSuite_VTK/'
    nameSimulation = 'sateNo4_1_'
    # Dimensions of the simulation
    numberDimensions = 3
    field_name = 'Uall'
    # If the field to be extracted is a velocity field --> velocityField = 1, otherwise 0
    velocityField = 0

    # Interval within the simulation for extracting data
    start = 40
    end = 99
    # Interval within the extracted data to perform PCA
    start_pca = 40
    end_pca = 99
    # Number of nodes in the unstructured mesh
    nsize = 851101#1244812

    extractFieldsAndPCA = extractFieldsAndPCA(directory_data=directory_data,
                nameSimulation=nameSimulation,
                field_name=field_name,
                velocityField = velocityField,
                start=start,
                end=end,
                start_pca=start_pca,
                end_pca=end_pca,
                nsize=nsize)

    # Extracts data
    #extractFieldsAndPCA.extractFields()

    # PCA on velocity fields
    extractFieldsAndPCA.PCA(field_name=field_name, velocityField=velocityField, numberDimensions=numberDimensions)

#Joining variables
#Two experiments

variableNames = ['Uall', 'nut']
#variableNames = ['u', 'v', 'w', 'nut']

pcs = np.zeros((59, 0))
for varName in variableNames:
    print(varName)
    tempPcs = np.load(f'{directory_data}sateNo4_1_pcs_{varName}_data_40_to_99.npy')
    print(tempPcs.shape)
    pcs = np.hstack((pcs, tempPcs))
    print(pcs.shape)

np.save(f'{directory_data}sateNo4_1_pcs_Uallnutsep_data_40_to_99.npy', pcs)

