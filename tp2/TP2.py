#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:22:09 2017

@author: laura
"""
#    df = pd.DataFrame(mdata)
#    df = pd.DataFrame(np.concatenate([ndata[c] for c in columns], axis=1),
#                  index=[datetime(*ts) for ts in ndata['timestamps']],
#                  columns=columns)


import pandas as pd
import numpy as np
import scipy
from scipy import signal 
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
from scipy.io import whosmat
from datetime import datetime, date, time
import seaborn

#P1='/home/laura/Documents/UBA/DataScience/TP2/P01.mat'
P1 = '/media/nbarbeito/DISK_IMG/TP2/P01.mat'

def Mat2Data(filename):
    mat = loadmat(filename)  # load mat-file    
    mdata = mat['data']  # variable in mat file
    return mdata

def calcular_media(paciente, epoch):
	return np.mean([paciente[epoch][7], 
				  paciente[epoch][43],
				  paciente[epoch][79],
				  paciente[epoch][130], 
				  paciente[epoch][184]], axis=0)

def plot_media(paciente, epoch):
	plt.plot(paciente[0][7],linestyle='--') # imprimir serie de tiempo
	plt.plot(paciente[epoch][43],linestyle='--')
	plt.plot(paciente[epoch][79],linestyle='--')
	plt.plot(paciente[epoch][130],linestyle='--')
	plt.plot(paciente[epoch][184],linestyle='--')
	plt.plot(calcular_media(paciente, epoch),'k')
	plt.show()

def analisis_espectral(paciente, epoch):
	return scipy.signal.welch(calcular_media(paciente, epoch), fs=250, nfft=2048)

def a1(paciente):
	z=[]
	"""for epoch in range(0, 10):
		freq, pot = analisis_espectral(paciente, epoch)
		z.append(pot)"""
	
	for epoch in range(0, 10):
		freq, pot = analisis_espectral(paciente, epoch)	
		plt.plot(freq, pot)
	plt.show()
	
	seaborn.heatmap(np.array(z).T)
	
	#plt.xlim(0, 50)
	plt.show()
		
		

p01=Mat2Data(P1)
#plt.plot(p01[0][7])
#plot_media(p01, 0)
#freq, pot = analisis_espectral(p01, 0)
#plt.plot(freq, pot)
#plt.xlim(0,50)
#plt.show()
a1(p01)
