#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn
from scipy import stats
from scipy.io import loadmat
import glob


def Mat2Data(filename):
    """ Lee los datos desde filename (.mat) a un np array """
    
    mat = loadmat(filename)  # load mat-file    
    mdata = mat['data']  # variable in mat file
    return mdata

def fix_max(x, m):
    """ Funcion para poner el maximo de la serie en el
    bin que le corresponde """
    if int(x) == m:
        return m - 1
    else:
        return x

def symb(serie):
    """ Dada una serie continua devuelve una 
    representación en símbolos y el simbolo mayor. 
    Cada simbolo es un entero mayor o igual a cero.
    Usa la regla de Scott """

    # Computo el N
    std = np.std(serie)
    maximo = np.amax(serie)
    minimo = np.amin(serie)
    t = np.power(len(serie), -1/3)
    #print([std, maximo, minimo, t])
    N = np.ceil((maximo - minimo) / (3.5*std*t))
    
    # Transformo la serie en simbolos
    step = np.abs(maximo - minimo)/N
    simb = np.floor(np.divide(np.subtract(serie, np.amin(serie)), step))

    return int(N), np.array([fix_max(s, int(N)) for s in simb])


def prob(serie):
    """ Dada una serie de numeros, devuelve la distribucion
    de los simbolos que la representan.
    """
    
    bins, simbolos = symb(serie)
    if bins == 0:
        print("BINS DIO CERO!")
    if bins - 1 != np.amax(simbolos):
        print("PROBLEMAS!!")


    hist = [0] * bins
    for s in simbolos:
        hist[int(s)] = hist[int(s)] + 1

    # Se puede reemplazar el np.sum(hist) por 201
    return np.divide(hist, np.sum(hist))
        

def entropia(serie):
    """ Calcula la entropia de la señal serie, medida en bits """

    p = prob(serie)
    p = p[p>0]
    return -np.inner(p, np.log2(p))


def entropia_bern():
    """ Testeo para entropia para una variable 
    bernoulli"""
    ps = np.arange(0.01, 1, 0.1)

    ent = []
    for p in ps:
        sample = []
        for i in range(0, 1000):
            r = np.random.random()
            if r < p:
                r = 0
            else:
                r = 1

            sample.append(r)
        if not 0 in sample or not 1 in sample:
            print("CERO CERO!")
        ent.append(entropia(sample))

    plt.plot(ps, ent)
    plt.show()

"""
p = Mat2Data('/home/nico/Descargas/datos EEG/P01.mat')

es = []
epoch = 0
for electrodo in range(0, len(p[epoch])):
    es.append(entropia(p[epoch][electrodo]))

plt.plot(es)
plt.show()

# Por alguna razón algunos electrodos tienen mas informacion
# ¿Lo podriamos agregar al informe?
plt.title("Electrodo 225")
plt.plot(p[0][224])
plt.show()

plt.title("Electrodo 256")
plt.plot(p[0][255])
plt.show()
"""

def calcular_media(sujeto, epoch):
    """ Calcula la media entre los electrodos 8, 44, 80, 131 y 185,
    para un sujeto y epoch.
    
    sujeto : numpy array
    """
    
    return np.mean([sujeto[epoch][7],
                    sujeto[epoch][43],
                    sujeto[epoch][79],
                    sujeto[epoch][130],
                    sujeto[epoch][184]], axis=0)

files=glob.glob('/home/nico/Descargas/datos EEG/*.mat')
entropias = {'S': [], 'P': []}

for file in files:
    print(file)
    sujeto = Mat2Data(file)
    grupo_sujeto = file[-7]

    # computo la señal promedio
    a=[]
    for epoch in range(0, len(sujeto)):
        a.append(np.mean(sujeto[epoch],axis=0)) # promedio electrodos    
    serie_prom = np.mean(a,axis=0) # promedio epochs

    entropias[grupo_sujeto].append(entropia(serie_prom))

seaborn.swarmplot(data=pd.DataFrame.from_dict(entropias))
plt.xlabel('Grupo')
plt.ylabel('Entropía en bits')
plt.show()

print("Test shapiro, entropía para grupo S"+str(stats.shapiro(entropias['S'])))
print("Test shapiro, entropía para grupo P"+str(stats.shapiro(entropias['P'])))

_, pv = stats.mannwhitneyu(entropias['S'], entropias['P'])
print("Test MannW-hitney-U grupo S: " + str(pv))

_, pv = stats.mannwhitneyu(entropias['S'], entropias['P'])
print("Test MannW-hitney-U grupo P: " + str(pv))

_, pv = stats.ranksums(entropias['S'], entropias['P'])
print("Test ranksums grupo S: " + str(pv))

_, pv = stats.ranksums(entropias['S'], entropias['P'])
print("Test ranksums grupo P: " + str(pv))

_, pv = stats.ttest_rel(entropias['S'], entropias['P'])
print(pv)
