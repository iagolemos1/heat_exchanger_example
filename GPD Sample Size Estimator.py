#------Importing packages------#
import numpy as np
import matplotlib.pyplot as plt
from thresholdmodeling import thresh_modeling
import random
import warnings
import math
import seaborn as sns

sns.set_style('ticks')
#------------------------------

#----Taking out warnings----#
warnings.simplefilter("ignore")
#----------------------------

#----Using thresholdmodeling package function to return just the shape, scale and sample over the threshold-------#
def fit_gpd(sample, threshold, fit_method):
    [shape, scale, sample_2, sample_excess, sample_over_thresh] = thresh_modeling.gpdfit(sample, threshold, fit_method)
    return(shape, scale, sample_over_thresh)
#------------------------------------------------------------------------------------------------------------------

#---------Function to extrapolate the observations for the non observerd area (Tan, 2017)-----------#
def GPD_extrapolation(sample_data, sample_data_fit, beamtubes, rm, length, shape, scale, ureal):
    surface_total_area = beamtubes*math.pi*2*rm*length #taking total surface of tubes
    surface_sampled = len(sample_data)*math.pi*2*rm*length #taking observed surface 
    lamb = len(sample_data_fit)/surface_sampled #taking rate of corrosion (mm/m^2)
    uninspec_tubes = lamb*(surface_total_area-surface_sampled) #taking the number of excesses in the uninspected tubes
    p2 = 1/uninspec_tubes #taking the probability related to the return period for the uninspected area
    c_ext = ureal + (scale/shape)*((p2**(-shape)) - 1) #taking the extrapolation 
    return(c_ext) 
#----------------------------------------------------------------------------------------------------    

#----Function to compare the original sample and the second sample and return the real maximum corrosion----#
def return_c_max_real(sample, resample):
    c = [sample.index(i) for i in resample] #getting indexes in the original sample array that are equal to the elements in the resample
    a = sample.copy() #turning into a new array the original one
    for element in sorted(c, reverse = True):  
        del a[element] #deleting the elements with the indexes
    c_max_real = max(a)
    return(c_max_real)
#------------------------------------------------------------------------------------------------------------

#----Function to cluster and re-sample the original sample by means of the contribution of each family----#
def clust_resample(sample, sample_size_req):

    clust_family = [] #cluster families
    clust_ocur = [] #how much each family appears 
    clust_weight = [] #weight of each family

    k = 0
    
    #---Algorithm to full the previous arrays---#
    sample.append(0)
    for i in range(1, len(sample)):
        if sample[i] != sample[i-1]:
            clust_family.append(sample[i-1])
            clust_ocur.append(i-k) 
            clust_weight.append((i-k)/len(sample))
            k = i
    #-------------------------------------------#

    factor = 0.5 #How much we start decreasing the number of each family appears (the half)
    new_sample_size = [float('inf')]
    
    #---Algorithm to resample---
    while sum(new_sample_size)>sample_size_req:
        new_sample_size = []
        for i in range(len(clust_family)):
            if clust_weight[i] < cl:
                new_sample_size.append(math.ceil(factor*clust_ocur[i]))
            elif clust_weight[i] > cl:
                new_sample_size.append(math.floor(factor*clust_ocur[i]))
        factor = factor - 0.00001
   
    new_sample = []
    new_sample_2 = []
    
    for i in range(len(clust_family)):
        new_sample.append(np.ones(new_sample_size[i])*clust_family[i])

    new_sample_2 = np.concatenate(new_sample, axis=0)
    #-----------------------------

    return(new_sample_2)
#-----------------------------------------------------------------------------------------

#----Main function to get the new sample size----
def main(sample_original, beamtubes, rm, length, threshold, fit_method):
    #Main Algorithm
    c_max_real = 0
    c_ext = 1
    n = 110
    while abs(c_max_real - c_ext)>=0.1:
        sample_data = sample_original.copy()
        resample = sorted(clust_resample(sample_data, n))
        c_max_real = max(sample_data)
        [shape, scale, sample_over_thresh] = fit_gpd(resample, threshold, fit_method)
        c_ext = GPD_extrapolation(resample, sample_over_thresh, beamtubes, rm, length, shape, scale, threshold)
        print(c_max_real)
        print(c_ext)
        n = n + 1
    print('-------Resultados-------')
    print('Parâmetros do trocador:\nEspessura do tubo: {} mm.\nTamanho da população: {} tubos.\nComprimento dos tubos: {} mm.\nDiâmetro dos tubos: {} mm.\nErro: \u00B1 {} mm.\nLimiar ótimo: {} mm.\nNível de confiança: {}%. \nTamanho amostral estimado: {} observações.'.format(width, beamtubes, length, diameter, 0.05, threshold, 95, n - 1))
    print('------------------------')
#------------------------------------------------

#Getting Data and treating
width = 2.10
filename = 'IRIS_data.txt'
sample_pure = np.array(np.loadtxt(fname=filename))
sample_original = sorted(np.array(width - sample_pure))


#Heat exchangers definitions
beamtubes = 500
length = 5200  
diameter = 25.4
rm = (diameter-(width/2))/2


#thresh_modeling.MRL(sample_original, 0.05)
#thresh_modeling.Parameter_Stability_plot(sample_original, 0.05)

#Statistical Definitions

fit_method = 'mle'
cl = 0.05

main(sample_original, beamtubes, rm, length, 0.5, fit_method)


