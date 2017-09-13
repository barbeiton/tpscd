#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn
from scipy.signal import welch
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
from scipy.io import whosmat

"Path a los datos de un paciente"
#P1='/home/laura/Documents/UBA/DataScience/TP2/P01.mat'
#P1 = '/media/nbarbeito/DISK_IMG/TP2/P01.mat'
P = '/home/nico/Descargas/S02.mat'
S = '/home/nico/Descargas/P02.mat'


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

    for electrodo in [7, 43, 79, 130, 184]:
        plt.plot(paciente[epoch][electrodo], linestyle='--')
    
    plt.plot(calcular_media(paciente, epoch),'k')
    
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('V^2')
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
    
    plt.xlim([0, 50])
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('V^2')
    plt.show()


def a1(paciente):
    """ Resolución del ejercicio a.1: heatmap de un paciente """

    z=[]
    for epoch in range(0, len(paciente)):
        _, pot = analisis_espectral(paciente, epoch)
        z.append(pot[0:400]) # No hay necesidad de ver las frecuencias mayores a 47Hz

    # ¿Cómo agregar bien los ticks en y?
    # ¿Cómo agregar V^2 al colormap, el coso a la derecha?
    seaborn.heatmap(np.array(z).T, cmap="YlGnBu_r", xticklabels=100, yticklabels=False)
    plt.xlabel('Epoch')
    plt.ylabel('Frequencia 50 <-> 0Hz')
    plt.show()
       



p = Mat2Data(P)
plot_media(p, 0)
plot_epocs(p)
a1(p)
