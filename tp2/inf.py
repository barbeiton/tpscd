#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import sys
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


def entropia_conjunta(serie1, serie2):
    """ Calcula la entropía conjunta de las dos
    series parámetro """

    bins1, s1 = symb(serie1)
    bins2, s2 = symb(serie2)
    
    hist = {}
    for ss1 in s1:
        for ss2 in s2:
            if str(ss1)+'-'+str(ss2) in hist:
                hist[str(ss1)+'-'+str(ss2)] = hist[str(ss1)+'-'+str(ss2)] + 1
            else:
                hist[str(ss1)+'-'+str(ss2)] = 1

    values = list(hist.values())
    probs = np.divide(values, np.sum(values))
    return -np.inner(probs, np.log2(probs))


def ej_2_2(file_path):
    """ Resolución del ejercicio 2.2 """
    
    files=glob.glob(file_path)
    entropias = {'S': [], 'P': []}
    entropias_conj = {'S': [], 'P': []}

    for file in files:
        print(file)
        sujeto = Mat2Data(file)
        grupo_sujeto = file[-7]
        
        a=[]
        b=[]
        for epoch in range(0, len(sujeto)):
            b.append(entropia_conjunta(sujeto[epoch][7], sujeto[epoch][43]))
            for electrodo in range(0, len(sujeto[epoch])):
                a.append(entropia(sujeto[epoch][electrodo]))
            
        entropias[grupo_sujeto].append(np.mean(a))
        entropias_conj[grupo_sujeto].append(np.mean(b))


    print("Tests para ejercicio 2.2.a")
    
    _, pv = stats.mannwhitneyu(entropias['S'], entropias['P'])
    print("Test MannW-hitney-U: " + str(pv))
    
    _, pv = stats.ranksums(entropias['S'], entropias['P'])
    print("Test ranksums: " + str(pv))

    _, pv = stats.ttest_ind(entropias['S'], entropias['P'])
    print("Test t-test: " + str(pv))
    
    seaborn.swarmplot(data=pd.DataFrame.from_dict(entropias))
    plt.title('Entropía promedio de sujetos en ambos grupos')
    plt.xlabel('Grupo')
    plt.ylabel('Entropía en bits')
    plt.show()

    print("*****************************")
    print("Tests para ejercicio 2.2.b")
    
    _, pv = stats.mannwhitneyu(entropias_conj['S'], entropias_conj['P'])
    print("Test MannW-hitney-U: " + str(pv))
    
    _, pv = stats.ranksums(entropias_conj['S'], entropias_conj['P'])
    print("Test ranksums: " + str(pv))

    _, pv = stats.ttest_ind(entropias_conj['S'], entropias_conj['P'])
    print("Test t-test: " + str(pv))
    
    seaborn.swarmplot(data=pd.DataFrame.from_dict(entropias_conj))
    plt.title('Entropía conjunta promedio de sujetos en ambos grupos')
    plt.xlabel('Grupo')
    plt.ylabel('Entropía conjunta en bits')
    plt.show()


if len(sys.argv) < 2:
    print("Para ejecutar el código usar")
    print("python3 inf.py dir")
    print("Donde dir es el path que contiene los EEG de los sujetos")
    sys.exit()

path = sys.argv[1]
if path[-1] != '/':
    path = path+'/'
    
path = path+'*.mat'

ej_2_2(path)




