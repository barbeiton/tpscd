#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn
from scipy import stats
from scipy.io import loadmat


def Mat2Data(filename):
    """ Lee los datos desde filename (.mat) a un np array """
    
    mat = loadmat(filename)  # load mat-file    
    mdata = mat['data']  # variable in mat file
    return mdata


def symb(serie):
    """ Dada una serie continua devuelve una 
    representación en símbolos. Usa la regla de Scott """

    # Computo el N
    std = np.std(serie)
    maximo = np.amax(serie)
    minimo = np.amin(serie)
    t = np.power(len(serie), -1/3)
    N = np.ceil((maximo - minimo) / (3.5*std*t))
    
    # Transformo la serie en simbolos
    step = np.abs(maximo - minimo)/N
    return np.ceil(np.divide(np.subtract(serie, np.amin(serie)), step))



p = Mat2Data('/home/nico/Descargas/datos EEG/P01.mat')
#print(p[0][0])
print(symb(p[0][0]))
plt.plot(p[0][0])
plt.show()

