#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from scipy import stats
from scipy.signal import welch
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import glob 


"""
P = '/home/laura/Documents/UBA/DataScience/TP2/S03.mat'
#path = '/home/laura/Documents/UBA/DataScience/TP2/*.mat'
"""


"""
S01 = '/media/nbarbeito/DISK_IMG/TP2/S01.mat'
path = '/media/nbarbeito/DISK_IMG/TP2/*.mat'
"""


P02 = '/home/nico/Descargas/datos EEG/P02.mat'
P03 = '/home/nico/Descargas/datos EEG/P03.mat'
S01 = '/home/nico/Descargas/datos EEG/S01.mat'
S02 = '/home/nico/Descargas/datos EEG/S02.mat'
path = '/home/nico/Descargas/datos EEG/*.mat'


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


def analisis_espectral(serie):
    """ Transformada de Welch para la serie de tiempo """

    return welch(serie, fs=250, nfft=2048, nperseg=201)


def plot_epocs(paciente):
    """ Grafica todos los epocs de un paciente,
        en el dominio de frecuencia promediando
        los electrodos 7, 43, 130, 184 """

    for epoch in range(0, len(paciente)):
        freq, pot = analisis_espectral(calcular_media(paciente, epoch))
        plt.plot(freq, pot)
    
    #plt.xlim([0, 45])
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel(r'$\frac{V^2}{Hz}$')
    plt.show()


def a1(paciente):
    """ Resolución del ejercicio a.1: heatmap de un paciente """

    z=[]
    for epoch in range(0, len(paciente)):
        freq, pot = analisis_espectral(calcular_media(paciente, epoch))
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
    
    z=[]
    for electrodo in range(0, len(paciente[0])):
        z=[]
        for epoch in range(0, len(paciente)):
            freq, pot = analisis_espectral(paciente[epoch][electrodo])
            z.append(pot)
        plt.plot(freq, np.mean(z,axis=0))

    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel(r'$\frac{V^2}{Hz}$')
    plt.xlim(0,47)
    plt.show()        


def b1(paciente):
    """ Para un paciente hacer el analisis espectral
    para la señal promedio entre electrodos y epochs"""
    
    a=[]
    for epoch in range(0, len(paciente)):
        a.append(np.mean(paciente[epoch],axis=0)) # promedio electrodos

    b = np.mean(a,axis=0) # promedio epochs
    freq,pot = analisis_espectral(b)
    
    return freq,pot


def b(path):
    """ Resolucion del ejercicio b """

    files=glob.glob(path)
    
    for file in files:
        print(file)
        freq,pot=b1(Mat2Data(file))
        plt.plot(freq,pot) # muestra los resultados de la fft para cada paciente
       
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel(r'$\frac{V^2}{Hz}$')
    plt.xlim(0,47)
    plt.show()


def filtrar(paciente, min, max, escala=1):
    """ Dado un paciente devuelve la potencia de la señal
    del promedio entre epocs y electrodos en la banda 
    [min, max) """

    freq, pot = b1(paciente)
    banda_pot = [escala * p[1] for p in zip(freq, pot) if p[0] < max and p[0] >= min]
    return banda_pot


def c_plot(paciente, min, max):

    banda_pot = filtrar(paciente, min, max, 10e18)
    seaborn.swarmplot(data=banda_pot)
    plt.show()


def c(path, log_scale=True):
    """ Resuelve el ejercicio c """
    
    # Si no se usa log_scale, hay que tunear la escala en el filter
    # (10e18) o setear bien el y lim que esta comentado abajo
    
    files=sorted(glob.glob(path))
    
    alpha = dict()
    for file in files:
        print(file)
        paciente = Mat2Data(file)
        nombre_paciente = file[-7:-4]

        if log_scale:
            alpha[nombre_paciente] = np.log10(filtrar(paciente, 8, 13))

        else:
            alpha[nombre_paciente] = filtrar(paciente, 8, 13)

    data = pd.DataFrame.from_dict(alpha)
    seaborn.swarmplot(data=data)
    #seaborn.violinplot(data=data) # si se descomenta hace ambos
    if not log_scale:
        plt.ylim(0, 10e-18)
    plt.show()


def p_total(paciente, min, max, escala=1):
    """ Devuelve la potencia total en la banda [min, max) """
    
    return np.sum(filtrar(paciente, min, max, escala))


def d(path, log_scale=True):
    """ Resuelve el ejercicio d """
    # Si no se usa log_scale, hay que tunear la escala en el filter
    # (10e18) o setear bien el y lim que esta comentado abajo
    
    cols = ['delta', 'theta', 'alpha', 'beta', 'gamma'] 
    banda = pd.DataFrame(columns=cols, index=range(1,21))
    
    files = glob.glob(path)
    i = 0
    for file in files:
        print(file)
        paciente = Mat2Data(file)

        delta = p_total(paciente, 0, 4.0)
        theta = p_total(paciente, 4.0, 8.0)
        alpha = p_total(paciente, 8.0, 13.0)
        beta = p_total(paciente, 13.0, 30.0)
        gamma = p_total(paciente, 30.0, 125.0)

        if log_scale:
            banda.loc[i] = np.log10([delta, theta, alpha, beta, gamma])

        else:
            banda.loc[i] = [delta, theta, alpha, beta, gamma]

        i = i + 1

    seaborn.swarmplot(data=banda)
    if not log_scale:
        plt.ylim(0, 10e-18)
    plt.show()


def e(path, log_scale=True):
    """ Resuelve el ejercicio e """

    # Si no se usa log_scale, hay que tunear la escala en el filter
    # (10e18) o setear bien el y lim que esta comentado abajo
    
    cols = ['delta', 'theta', 'alpha', 'beta', 'gamma'] 
    banda = pd.DataFrame(columns=cols, index=range(1,5))
    
    files = glob.glob(path)
    i = 1
    for file in files:
        print(file)
        paciente = Mat2Data(file)

        delta = p_total(paciente, 0, 4.0) / 4.0
        theta = p_total(paciente, 4.0, 8.0) / 4.0
        alpha = p_total(paciente, 8.0, 13.0) / 5.0
        beta = p_total(paciente, 13.0, 30.0) / 17.0
        gamma = p_total(paciente, 30.0, 125.0) / 15.0

        if log_scale:
            banda.loc[i] = np.log10([delta, theta, alpha, beta, gamma])

        else:
            banda.loc[i] = [delta, theta, alpha, beta, gamma]

        i = i + 1

    """
    _, pv = stats.f_oneway(banda['delta'], banda['theta'], banda['alpha'], banda['beta'], banda['gamma'])
    print(pv)
    """

    seaborn.swarmplot(data=banda)
    if not log_scale:
        plt.ylim(0, 10e-18)
    plt.show()

p = Mat2Data(S02)
#plot_media(p, 0)
#plot_epocs(p)


#a1(p)
#a2(p)
#b(path)
#c(path)
d(path)
#e(path)
