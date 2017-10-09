
# coding: utf-8

# ##### Los features que vamos a utilizar son:
# 
# Potencia para cada banda de frecuencia (Delta, Theta, Alpha, Beta y Gamma)

# Potencia normalizada para las mismas bandas de frecuencia.

# In[ ]:




# In[1]:

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


# In[ ]:

def analisis_espectral(serie):
    """ Transformada de Welch para la serie de tiempo serie"""

    return welch(serie, fs=250, nfft=2048, nperseg=201)


# In[549]:

#TP2
def Mat2Data(filename):
    """ Lee los datos desde filename (.mat) a un np array """
    
    mat = loadmat(filename)  # load mat-file    
    mdata = mat['data']  # variable in mat file
    return mdata


# In[ ]:




# In[10]:

#TP2
# Calculo la potencia total del sujeto en cada banda
def p_total(sujeto, min, max, escala=1):
    """ Devuelve la potencia total en la banda [min, max) """
    
    return np.sum(filtrar(sujeto, min, max, escala))


# In[12]:

#TP2 b) Calcular los valores de cada banda de frecuencia, promediados entre los electrodos (todos) y epochs para cada sujeto.
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


# In[13]:

#TP2
def filtrar(sujeto, min, max, escala=1):
    """ Dado un sujeto devuelve la potencia de la señal
    del promedio entre epocs y electrodos en la banda 
    [min, max)
    
    sujeto : numpy array
    """

    freq, pot = b1(sujeto)
    banda_pot = [escala * p[1] for p in zip(freq, pot) if p[0] < max and p[0] >= min]
    return banda_pot

#TP2
delta = p_total(sujeto, 0, 4.0)
theta = p_total(sujeto, 4.0, 8.0)
alpha = p_total(sujeto, 8.0, 13.0)
beta = p_total(sujeto, 13.0, 30.0)
gamma = p_total(sujeto, 30.0, 125.0)


# In[14]:

#solo para ver 
cols = ['delta', 'theta', 'alpha', 'beta', 'gamma']
print(' | '.join(cols))
potencia = [delta, theta, alpha, beta, gamma]
print(' | '.join(map(str, potencia)))


# In[2]:

##TP Definiciones
bandasRango = {"delta": (0,4.0), "theta": (4.0,8.0), "alpha": (8.0,13.0), "beta": (13.0,30.0), "gamma": (30.0,126.0)}
bandas = bandasRango.keys()


# In[3]:

##TP Definiciones
labels=['P','S']
pacientes=[]
for l in labels:
    pacientes += [l + "{:02d}".format(i) for i in range(1,  11)]


# In[4]:

columnas = np.concatenate((
    ['name'],
    list(map(lambda n: n+'_m', bandas)), 
    list(map(lambda n: n+'_norm_m', bandas)), 
    list(map(lambda n: n+'_std', bandas)),
    list(map(lambda n: n+'_norm_std', bandas)),
    ['label']
))
columnas


# In[584]:

outFile = 'features'
df = pd.DataFrame(columns=columnas)
df.to_csv('./features/'+outFile+'.csv', columns=columnas, index =False)



# In[573]:

##Definiciones Auxiliares
def bandaID(freq):
    for k, (_, v) in enumerate(bandasRango.items()):
        if freq >= v[0] and freq < v[1]:
            return k
          


# In[574]:

def dictBandas(inicial):
    res = []
    for k in range(0, len(bandas)):
        res.append(inicial)
    return res

#def dictBandas(inicial):
#    res = {}
#    for k in bandas:        
#        res[k]=inicial
#    return res

 #for k, v in enumerate(bandasRango.items()):
 #   print(k, v)


# In[575]:

def load(name):
    """ Lee los datos desde filename (.mat) a un np array """    
    sujeto_file = './DataSet/'+name+'.mat' #file path de un sujeto    
    return Mat2Data(sujeto_file)


# In[586]:

# TODO consultar: TP2 b) Calcular los valores de cada banda de frecuencia, promediados entre los electrodos (todos) 
# y epochs para cada sujeto.
# Entiendo que debe ser las medias del analisis espectral en vez de el análisis espectral de las medias (b1)
# El cálculo del poder espectral en cada banda se realiza sumando los valores obtenidos de la FFT que corresponden a las
# frecuencias incluídas en la misma.
def poderEspectral(sujeto):
    poderEpochs = dictBandas([])
    poderNormalizado = dictBandas([])
    for epoch in range(0, len(sujeto)):#len(sujeto)
        poderElectrodos = dictBandas([])
        for electrodo in range(0, len(sujeto[epoch])):
            freq, pot = analisis_espectral(sujeto[epoch][electrodo])                
            poderFreq = dictBandas(0)
            for f, p in zip(freq, pot):
            
                poderFreq[bandaID(f)] += p 
            
            for banda in range(0, len(bandas)):
                poderElectrodos[banda].append(poderFreq[banda])
        
        suma = 0
        for banda in range(0, len(bandas)):            
            poder = np.mean(poderElectrodos[banda])
            poderEpochs[banda].append(poder)
            suma+=poder
        for banda in range(0, len(bandas)):            
            poderNormalizado[banda].append(poderEpochs[banda]/suma)
    return poderEpochs, poderNormalizado


# In[587]:

def mainFeatures(outFile):
    data = pd.DataFrame({})
    for paciente in pacientes:
        sujeto = load(paciente)
        poderEspectralEpochs, poderNormalizadoEpochs = poderEspectral(sujeto)
        
        media = np.mean(poderEspectralEpochs, axis=1)
        media_normalizados = np.mean(poderEspectralEpochs, axis=1)
        desvio = np.std(poderEspectralEpochs, axis=1)
        desvio_normalizado = np.std(poderEspectralEpochs, axis=1)
        
        serie = pd.Series(np.concatenate(([paciente],media, media_normalizados, desvio, desvio_normalizado, [paciente[0]]), axis=0))
                         
        #save to csv
        df = pd.DataFrame({})
        df=df.append(serie, ignore_index=True)        
        df.to_csv('./features/'+outFile +'.csv', mode='a', header=False, index =False)        
        df.to_csv('./features/'+outFile + paciente + '.csv', mode='w', index =False)
        
        data = pd.concat([data, df], ignore_index=True)
    return data


# In[ ]:

mainFeatures('features')


# In[ ]:

#df = mainFeatures('features')


# In[478]:




# In[5]:

def plotROC(xs, ys, name):
    y_true = label_binarize(ys, classes=['P', 'S']).ravel()
    y_score = xs

    # Compute ROC curve and ROC area for each class     
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    plt.title('ROC Curve: ' + name)
    plt.legend(loc="lower right")
    plt.show()


# In[6]:

def featuresROC(df, ys):
    
    for k in range(1,20):
        name = columnas[k]
        xs = np.concatenate(df.loc[:,[name]].values)
        plotROC(xs, ys, name)


# In[9]:

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


# In[7]:

df = pd.read_csv('./features/features.csv')


# In[31]:

#TODO faltan procesar features. Esto es una prueba
#ys =['P']*10+['S']*10
ys =['P']*8+['S']*2
featuresROC(df, ys)


# In[24]:

def featuresCVROC(df, ys):
    
    for k in range(1,20):
        name = columnas[k]
        xs = np.concatenate(df.loc[:,[name]].values)
        plotCVROC(xs, ys, name)
   


# In[25]:

#TODO consultar
def plotCVROC(xs, ys, name):
    y_true = label_binarize(ys, classes=['P', 'S']).ravel()
   
    # shuffle and split training and test sets
    xs_train, xs_test, y_train, y_test = train_test_split(xs, y_true, test_size=.5,
                                                    random_state=0)
    
    random_state = np.random.RandomState(0)
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
    
    y_score = classifier.fit(xs_train, y_train).decision_function(xs_test)

    # Compute ROC curve and ROC area for each class     
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    plt.title('ROC Curve: ' + name)
    plt.legend(loc="lower right")
    plt.show()


# In[33]:

#ys =['P']*10+['S']*10 TODO: Faltan procesar
ys =['P']*8+['S']*2
featuresCVROC(df, ys)


# In[ ]:




# In[ ]:




# Una medida de información intra-electrodo (a elección)

# Una medida de informacion inter-electrodo (a elección)
