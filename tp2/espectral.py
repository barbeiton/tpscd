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
        z.append(pot[0:400]) # No hay necesidad de ver las frecuencias mayores a 47Hz
    df = pd.DataFrame(data=np.array(z).T, index=np.around(freq[0:400], decimals=1))
    seaborn.heatmap(df, cmap="YlGnBu_r", xticklabels=100,yticklabels=20,cbar_kws={'label': r'$\frac{V^2}{Hz}$'})
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


def solo_algunos(sujeto):
    """ Para un sujeto hacer el analisis espectral
    para la señal promedio entre todos los  electrodos y las epochs
    
    sujeto : numpy array
    """
    
    a=[]
    for epoch in range(0, len(sujeto)):
        a.append(calcular_media(sujeto, epoch)) # promedio electrodos
        
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

def categorical_plot(data, log_scale=True):
    """ Hace todos los categorical plots de seaborn """

    seaborn.violinplot(data=data)

    """seaborn.stripplot(data=data)
    seaborn.boxplot(data=data)
    seaborn.violinplot(data=data)
    seaborn.lvplot(data=data)
    seaborn.pointplot(data=data)
    seaborn.barplot(data=data)
    seaborn.countplot(data=data)"""
    
    plt.xlabel("Sujetos")
    if log_scale:
        plt.ylabel("Potencia (log10)")
    plt.show()


def p_total(sujeto, min, max, escala=1):
    """ Devuelve la potencia total en la banda [min, max) """
    
    return np.sum(filtrar(sujeto, min, max, escala))

def test_shapiro_bandas(banda):
    shapiro_delta=stats.shapiro(banda['delta'])
    print("shapiro (delta) " + str(shapiro_delta))
    shapiro_theta=stats.shapiro(banda['theta'])
    print("shapiro (theta) " + str(shapiro_theta))
    shapiro_alpha=stats.shapiro(banda['alpha'])
    print("shapiro (alpha) " + str(shapiro_alpha))
    shapiro_beta=stats.shapiro(banda['beta'])
    print("shapiro (beta) " + str(shapiro_beta))
    shapiro_gamma=stats.shapiro(banda['gamma'])
    print("shapiro (gamma) " + str(shapiro_gamma))


def test_estadistico_entre_bandas(banda, test):
    """ Aplica los test elegidos a las bandas de frecuencia (banda),
    averigua si hay suficiente evidencia para afirmar que las medias son 
    distintas"""
    
    _, pv = test(banda['delta'], banda['theta'])
    print(" test (delta vs theta) " + str(pv))
    
    _, pv = test(banda['theta'], banda['alpha'])
    print("test (theta vs alpha) "+ str(pv))
    
    _, pv = test(banda['alpha'], banda['beta'])
    print("test (alpha vs beta) "+ str(pv))
    
    _, pv = test(banda['beta'], banda['gamma'])
    print("test (beta vs gamma) "+ str(pv))

    _, pv = test(banda['delta'], banda['gamma'])
    print("test (delta vs gamma) "+ str(pv))


def test_estadistico_entre_grupos(banda, test):
    """Aplica los test elegidos para comparar las medias de 
    distintas bandas entre los grupos"""

    grupo1 = banda.query('group==1')
    grupo2 = banda.query('group==2')

    _, pv = test(grupo1['delta'], grupo2['delta'])
    print("test (delta grupo1 vs delta grupo2) " + str(pv))

    _, pv = test(grupo1['theta'], grupo2['theta'])
    print("test (theta grupo1 vs theta grupo2) " + str(pv))

    _, pv = test(grupo1['alpha'], grupo2['alpha'])
    print("test (alpha grupo1 vs alpha grupo2) " + str(pv))

    _, pv = test(grupo1['beta'], grupo2['beta'])
    print("test (beta grupo1 vs beta grupo2) " + str(pv))

    _, pv = test(grupo1['gamma'], grupo2['gamma'])
    print("test (gamma grupo1 vs gamma grupo2) " + str(pv))


def estadisticos_puntos_dye(banda, banda_norm, test):
    """ Test estadísticos (puntos d y e) """

    print("Test estadístico para potencias no normalizadas")
    test_estadistico_entre_bandas(banda, test)
    
    print("Test estadístico para potencias normalizadas")
    test_estadistico_entre_bandas(banda_norm, test)

    print("Test estadístico entre grupos")
    test_estadistico_entre_grupos(banda, test)
    
    print("Test estadístico entre grupos, datos normalizados")
    test_estadistico_entre_grupos(banda_norm, test)


def escalar_log(banda):
    """ Funcion para aplicar logaritmo(10) para
    mejorar el grafico de los datos """

    for i in range(1, banda.shape[0] + 1):
        s = np.array(banda.loc[i].values, dtype=float)
        banda.loc[i] = np.log10(s)
    return banda


def graficar_puntos_dye(banda, banda_norm, log_scale):
    """ Graficos correspondientes a los puntos d y e """

    if log_scale:
#        seaborn.violinplot(x='variable', y='value', hue='group', data=banda,inner='point')
        seaborn.violinplot(data=escalar_log(banda).iloc[:, 0:5],inner='point')
        plt.ylabel("Potencia en escala log10")
    else:
        seaborn.violinplot(data=banda.iloc[:, 0:5],inner='point')
        plt.ylim(10e-19, 10e-16)
        plt.ylabel("Potencia")

    plt.xlabel("Banda")
    plt.show()

    if log_scale:
        seaborn.violinplot(data=escalar_log(banda_norm).iloc[:, 0:5],inner='point')
        plt.ylabel("Potencia en escala log10")
    else:
        seaborn.violinplot(data=banda_norm.iloc[:, 0:5],inner='point')
        plt.ylim(10e-19, 10e-16)
        plt.ylabel("Potencia")

    plt.xlabel("Banda")
    plt.show()


def cydye(path, log_scale=True):
    """ Resuelve el ejercicio c, d y el e
    Plot categórico de las potencias en las distintas bandas de frecuencia
    para cada paciente. Para potencias normalizadas y no normalizadas.
    Se aplican test estadísticos apropiados.
    path : directorio con archivos de EEG de sujetos
    log_scale : usar escala logarítmica base 10
    """
  
    print("RESOLVIENDO EJERCICIOS d Y e")

    files = sorted(glob.glob(path))

    cols = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'group'] 
    banda = pd.DataFrame(columns=cols, index=range(1, len(files) + 1))
    banda_norm = pd.DataFrame(columns=cols, index=range(1, len(files) + 1))
    banda_alpha = dict()
    
    i = 1
    for file in files:
        print(file)
        sujeto = Mat2Data(file)

        if file[-7] == 'S':
            group = 1
        else:
            group = 2
        
        nombre_sujeto = file[-7:-4]

        a = filtrar(sujeto, 8.0, 13.0)

        # Calculo la potencia total del sujeto en cada banda
        # (punto d)
        delta = p_total(sujeto, 0, 4.0)
        theta = p_total(sujeto, 4.0, 8.0)
        alpha = np.sum(a)
        beta = p_total(sujeto, 13.0, 30.0)
        gamma = p_total(sujeto, 30.0, 125.0)

        # Calculo la potencia total normalizada en cada banda del sujeto 
        # Divido por el ancho de banda de cada banda
        # (punto e)
        delta_norm = delta / 4.0
        theta_norm = theta / 4.0
        alpha_norm = alpha / 5.0
        beta_norm = beta / 17.0
        gamma_norm = gamma / 15.0 # filtro pone limite en 45hz
		
        banda.loc[i] = [delta, theta, alpha, beta, gamma, group]
        banda_norm.loc[i] = [delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm, group] 
 
        # punto c
        if log_scale:
            banda_alpha[nombre_sujeto] = np.log10(a)
        else:
            banda_alpha[nombre_sujeto] = filtrar(sujeto, 8.0, 13.0)
            
        i = i + 1
#    df=pd.melt(banda, value_vars=['delta', 'theta', 'alpha', 'beta', 'gamma'], id_vars='group')    
#    print(df)
    print("Test estadí­stico Shapiro para bandas no normalizadas")
    test_shapiro_bandas(banda)
    
    print("Test estadí­stico Shapiro para bandas normalizadas")
    test_shapiro_bandas(banda_norm)
    
    print("TEST: wilcoxon, todos los electrodos")
#    estadisticos_puntos_dye(banda, banda_norm, stats.wilcoxon)

    print("TEST: ranksums ,todos los electrodos")
    estadisticos_puntos_dye(banda, banda_norm, stats.ranksums)

    print("TEST: mannwhitneyu, todos los electrodos")
    estadisticos_puntos_dye(banda, banda_norm, stats.mannwhitneyu)

    # Grafico correspondiente al punto c
    df_banda_alpha = pd.DataFrame.from_dict(banda_alpha)
    categorical_plot(df_banda_alpha, log_scale)

    # Graficos de los puntos d y e
    graficar_puntos_dye(banda, banda_norm, log_scale)
        

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


    # sujeto_file = '' # descomentar para cancelar la ejecucion, borrarlo dsp
    # Para ejecutar esto agregar la opción -s seguido de un sujeto
    if sujeto_file != '':
        sujeto = Mat2Data(sujeto_file)

        #plot_media(sujeto, 0)
        #plot_epocs(sujeto)
        a1(sujeto_file)
        a2(sujeto_file)
        

    # path = '' # descomentar para cancelar la ejecucion, borrarlo dsp
    # Para ejecutar esto agregar la opción -S seguido por el directorio 
    # donde están los archivos del eeg
    if path != '':
        #b(path)        
        cydye(path)


""" **********  """
main()
