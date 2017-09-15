#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn
from scipy.signal import welch
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import glob 

"Path a los datos de un paciente"
#P='/home/laura/Documents/UBA/DataScience/TP2/S03.mat'
#P1 = '/media/nbarbeito/DISK_IMG/TP2/S01.mat'
#P = '/home/nico/Descargas/S02.mat'
#S = '/home/nico/Descargas/P02.mat'


def Mat2Data(filename):
    """ Lee los datos desde filename (.mat) a un np array """

    mat = loadmat(filename)  # load mat-file    
    mdata = mat['data']  # variable in mat file
    return mdata


def calcular_media(paciente, epoch):
    """ Calcula la media entre los electrodos 8, 44, 80, 131 y 185,
    para un paciente y epoch"""

    return np.mean([paciente[epoch][7], 
        paciente[epoch][43],
        paciente[epoch][79],
        paciente[epoch][130], 
        paciente[epoch][184]], axis=0)


def plot_media(paciente, epoch):
    """ Grafica la media entre los electrodos 8, 44, 80, 131 y 185 y
    los datos en dichos electrodos, para un paciente y epoch"""
    x=np.arange(0,801,4)
    for electrodo in [7, 43, 79, 130, 184]:
        plt.plot(x,paciente[epoch][electrodo], linestyle='--')
    
    plt.plot(x,calcular_media(paciente, epoch),'k')
    
    plt.xlabel('Tiempo(ms)')
    plt.ylabel('V')
    plt.show()


def analisis_espectral(paciente, epoch):
    """ Transformada de Welch para la señal media entre los electrodos 
    8, 44, 80, 131 y 185, para un paciente y un epoch """

    return welch(calcular_media(paciente, epoch), fs=250, nfft=2048)


def plot_epocs(paciente):
    """ Grafica todos los epocs de un paciente, 
        en el dominio de frecuencia """

    for epoch in range(0, len(paciente)):
        freq, pot = analisis_espectral(paciente, epoch)	
        plt.plot(freq, pot)
    
#    plt.xlim([0, 45])
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel(r'$\frac{V^2}{Hz}$')
    plt.show()


def a1(paciente):
    """ Resolución del ejercicio a.1: heatmap de un paciente """
    z=[]
    for epoch in range(0, len(paciente)):
        freq, pot = analisis_espectral(paciente, epoch)
        z.append(pot) # No hay necesidad de ver las frecuencias mayores a 47Hz
    yticks = list(freq)
    # ¿Cómo agregar bien los ticks en y?
    # ¿Cómo agregar V^2 al colormap, el coso a la derecha?
    seaborn.heatmap(np.array(z).T, cmap="YlGnBu_r", xticklabels=100)
    plt.xlabel('Epoch')
    plt.ylabel('Frequencia 50 <-> 0Hz')
    plt.show()
    
def a2(paciente):
    """Resolucion del ejercicio a.2"""
    #Entendiendo que cada canal es cada electrodo
    z=[]
    for electrodo in range(0, len(paciente[0])):
        z=[]
        for epoch in range(0, len(paciente)):
            freq, pot = welch(paciente[epoch][electrodo], fs=250, nfft=2048,noverlap=45)
            z.append(pot)
        plt.plot(freq, np.mean(z,axis=0))    
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel(r'$\frac{V^2}{Hz}$')
    plt.xlim(0,47)
    plt.show()        
       
def b(paciente):
    """Resolucion del ejercicio b"""
    a=[]
    for epoch in range(0, len(paciente)):
        a.append(np.mean(paciente[epoch],axis=0))#promedio electrodos
    b=np.mean(a,axis=0)#promedio epochs
    freq,pot= welch(b, fs=250, nfft=2048)
    return freq,pot

path = '/home/laura/Documents/UBA/DataScience/TP2/*.mat'
def b1(path):
    files=glob.glob(path)
    for file in files:
        print(file)
        freq,pot=b((Mat2Data(file)))
        plt.plot(freq,pot) #muestra los resultados de la fft para cada paciente
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel(r'$\frac{V^2}{Hz}$')
    plt.xlim(0,47)
    plt.show()        

        
    
#p = Mat2Data(P)
#plot_media(p, 0)
#plot_epocs(p)
#a1(p)
#a2(p)
b1(path)