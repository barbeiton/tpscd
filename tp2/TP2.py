#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import glob 
import sys
import getopt
from scipy import stats
from scipy.signal import welch
from scipy.io import loadmat  # this is the SciPy module that loads mat-files

""" *********** """

def uso():
    print("Para ejecutar el código:")
    print("  python3 TP2.py -S path_sujetos")
    print("  python3 TP2.py -s sujeto_file")
    print("  python3 TP2.py -S path_sujetos -s sujeto_file")
    print("Donde path_sujetos es el directorio con los datos y sujeto_file es un archivo de un sujeto")
    sys.exit()


def Mat2Data(filename):
    """ Lee los datos desde filename (.mat) a un np array """

    mat = loadmat(filename)  # load mat-file    
    mdata = mat['data']  # variable in mat file
    return mdata


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


def plot_media(sujeto, epoch):
    """ Grafica la media entre los electrodos 8, 44, 80, 131 y 185 y
    los datos en dichos electrodos, para un sujeto y epoch
    
    sujeto : numpy array
    """

    x=np.arange(0,801,4)
    for electrodo in [7, 43, 79, 130, 184]:
        plt.plot(x,sujeto[epoch][electrodo], linestyle='--')
    
    plt.plot(x,calcular_media(sujeto, epoch),'k')
    
    plt.xlabel('Tiempo(ms)')
    plt.ylabel('V')
    plt.show()


def plot_epocs(sujeto):
    """ Grafica todos los epocs de un sujeto,
        en el dominio de frecuencia promediando
        los electrodos 7, 43, 130, 184 
        
    sujeto : numpy array"""

    for epoch in range(0, len(sujeto)):
        freq, pot = analisis_espectral(calcular_media(sujeto, epoch))
        plt.plot(freq, pot)
    
    plt.xlim([0, 50])
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel(r'$\frac{V^2}{Hz}$')
    plt.show()


def analisis_espectral(serie):
    """ Transformada de Welch para la serie de tiempo serie"""

    return welch(serie, fs=250, nfft=2048, nperseg=201)


def a1(sujeto_file):
    """ Resolución del ejercicio a.1: heatmap de un sujeto 
    
    sujeto_file : file path de un sujeto
    """
    
    sujeto = Mat2Data(sujeto_file)
    
    z=[]
    for epoch in range(0, len(sujeto)):
        freq, pot = analisis_espectral(calcular_media(sujeto, epoch))
        z.append(pot[0:500]) # No hay necesidad de ver las frecuencias mayores a 47Hz

    seaborn.heatmap(np.array(z).T, cmap="YlGnBu_r", xticklabels=100)
    plt.xlabel('Epoch')
    plt.ylabel('Frequencia (HZ)')
    plt.show()


def a2(sujeto_file):
    """Resolucion del ejercicio a.2: 
    potencia media entre epochs para cada electrodo en función de la
    frecuencia

    sujeto_file : file path de un sujeto
    """
    
    sujeto = Mat2Data(sujeto_file)

    z=[]
    for electrodo in range(0, len(sujeto[0])):
        z=[]
        for epoch in range(0, len(sujeto)):
            freq, pot = analisis_espectral(sujeto[epoch][electrodo])
            z.append(pot)
        plt.plot(freq, np.mean(z,axis=0))

    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel(r'$\frac{V^2}{Hz}$')
    plt.xlim([0,45])
    plt.show()        


def b1(sujeto):
    """ Para un sujeto hacer el analisis espectral
    para la señal promedio entre todos los  electrodos y las epochs
    
    sujeto : numpy array
    """
    
    a=[]
    for epoch in range(0, len(sujeto)):
        a.append(np.mean(sujeto[epoch],axis=0)) # promedio electrodos

    b = np.mean(a,axis=0) # promedio epochs
    freq,pot = analisis_espectral(b)
    
    return freq,pot


def b(path):
    """ Resolucion del ejercicio b 
    
    
    """

    files=glob.glob(path)
    
    for file in files:
        print(file)
        freq,pot=b1(Mat2Data(file))
        plt.plot(freq,pot) # muestra los resultados de la fft para cada sujeto
       
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel(r'$\frac{V^2}{Hz}$')
    plt.xlim(0,50)
    plt.show()


def filtrar(sujeto, min, max, escala=1):
    """ Dado un sujeto devuelve la potencia de la señal
    del promedio entre epocs y electrodos en la banda 
    [min, max)
    
    sujeto : numpy array
    """

    freq, pot = b1(sujeto)
    banda_pot = [escala * p[1] for p in zip(freq, pot) if p[0] < max and p[0] >= min]
    return banda_pot


def c(path, log_scale=True):
    """ Resuelve el ejercicio c 
    
    Swarm plot de la banda alpha para cada sujeto

    path : directorio con archivos de EEG de sujetos
    log_scale : usar escala logarítmica base 10
    """
    
    # Si no se usa log_scale, hay que tunear la escala en el filter
    # (10e18) o setear bien el y lim que esta comentado abajo
    
    files=sorted(glob.glob(path))
    
    alpha = dict()
    for file in files:
        print(file)
        sujeto = Mat2Data(file)
        nombre_sujeto = file[-7:-4]

        if log_scale:
            alpha[nombre_sujeto] = np.log10(filtrar(sujeto, 8, 13))

        else:
            alpha[nombre_sujeto] = filtrar(sujeto, 8, 13)

    data = pd.DataFrame.from_dict(alpha)
    seaborn.swarmplot(data=data)
    #seaborn.violinplot(data=data) # si se descomenta superpone ambos

    if not log_scale:
        plt.ylim(0, 10e-18)

    plt.xlabel("Bandas de frecuencia")
    if log_scale:
        plt.ylabel("Potencia [log10]")
    else:
        plt.ylabel("Potencia")
    plt.show()


def p_total(sujeto, min, max, escala=1):
    """ Devuelve la potencia total en la banda [min, max) """
    
    return np.sum(filtrar(sujeto, min, max, escala))

def test_estadistico_entre_bandas(banda):
    """ Aplica los test elegidos a las bandas de frecuencia (banda),
    averigua si hay suficiente evidencia para afirmar que las medias son 
    distintas"""
    
    _, pv = stats.f_oneway(banda['delta'], banda['theta'], banda['alpha'], banda['beta'], banda['gamma'])
    print("ANOVA test: " + str(pv))

    _, pv = stats.ttest_rel(banda['delta'], banda['theta'])
    print("t-test (delta vs theta) " + str(pv))
    
    _, pv = stats.ttest_rel(banda['theta'], banda['alpha'])
    print("t-test (theta vs alpha) "+ str(pv))
    
    _, pv = stats.ttest_rel(banda['alpha'], banda['beta'])
    print("t-test (alpha vs beta) "+ str(pv))
    
    _, pv = stats.ttest_rel(banda['beta'], banda['gamma'])
    print("t-test (beta vs gamma) "+ str(pv))



def dye(path, log_scale=True):
    """ Resuelve el ejercicio d y el e
    Plot categórico de las potencias en las distintas bandas de frecuencia
    para cada paciente. Para potencias normalizadas y no normalizadas.
    Se aplican test estadísticos apropiados.

    path : directorio con archivos de EEG de sujetos
    log_scale : usar escala logarítmica base 10
    """
    # Si no se usa log_scale, hay que tunear la escala en el filter
    # (10e18) o setear bien el y lim que esta comentado abajo
  
    print("RESOLVIENDO EJERCICIOS d Y e")

    files = glob.glob(path)

    cols = ['delta', 'theta', 'alpha', 'beta', 'gamma'] 
    banda = pd.DataFrame(columns=cols, index=range(1, len(files) + 1))
    banda_norm = pd.DataFrame(columns=cols, index=range(1, len(files) + 1))
    
    i = 1
    for file in files:
        print(file)
        sujeto = Mat2Data(file)
        
        # Calculo la potencia total del sujeto en cada banda
        delta = p_total(sujeto, 0, 4.0)
        theta = p_total(sujeto, 4.0, 8.0)
        alpha = p_total(sujeto, 8.0, 13.0)
        beta = p_total(sujeto, 13.0, 30.0)
        gamma = p_total(sujeto, 30.0, 125.0)

        # Calculo la potencia total normalizada en cada banda del sujeto 
        delta_norm = delta / 4.0
        theta_norm = theta / 4.0
        alpha_norm = alpha / 5.0
        beta_norm = beta / 17.0
        gamma_norm = gamma / 15.0 # filtro pone limite en 45hz
        
        if log_scale:
            banda.loc[i] = np.log10([delta, theta, alpha, beta, gamma])
            banda_norm.loc[i] = np.log10([delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm])

        else:
            banda.loc[i] = [delta, theta, alpha, beta, gamma]
            banda_norm.loc[i] = [delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm]

        i = i + 1
   

    print("Test estadístico para potencias no normalizadas")
    test_estadistico_entre_bandas(banda)

    seaborn.swarmplot(data=banda)    
    if not log_scale:
        plt.ylim(0, 10e-18)
    plt.show()

    print("Test estadístico para potencias normalizadas")
    test_estadistico_entre_bandas(banda_norm)

    seaborn.swarmplot(data=banda_norm)
   
    if not log_scale:
        plt.ylim(0, 10e-18)
    plt.show()


def main():
    """ Procesa los argumentos de linea de comando 
    y ejecuta las funciones que resuelven los ejercicios"""
    
    path = ''
    sujeto_file = ''

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'S:s:', [])

    except getopt.GetoptError:
        uso()

    if len(opts) == 0:
        uso()
    
    for opt, arg in opts:
        if opt == '-S':
            if arg[-1] != '/':
                arg = arg+'/'
            path = arg+'*.mat'

        elif opt == '-s':
            sujeto_file = arg

        else:
            print("Opción no reconocida")


    sujeto_file = '' # descomentar para cancelar la ejecucion, borrarlo dsp
    if sujeto_file != '':
        sujeto = Mat2Data(sujeto_file)

        #plot_media(sujeto, 0)
        #plot_epocs(sujeto)
        a1(sujeto_file)
        a2(sujeto_file)
        

    #path = '' # descomentar para cancelar la ejecucion, borrarlo dsp
    if path != '':
        #b(path)        
        #c(path)
        dye(path)


""" **********  """
main()

