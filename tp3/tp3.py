
# coding: utf-8

# In[11]:



# ##### Los features que vamos a utilizar son:
# 
# Potencia para cada banda de frecuencia (Delta, Theta, Alpha, Beta y Gamma)

# Potencia normalizada para las mismas bandas de frecuencia.

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
from collections import OrderedDict

def analisis_espectral(serie):
    """ Transformada de Welch para la serie de tiempo serie"""

    return welch(serie, fs=250, nfft=2048, nperseg=201)


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


#TP2
def Mat2Data(filename):
    """ Lee los datos desde filename (.mat) a un np array """
    
    mat = loadmat(filename)  # load mat-file    
    mdata = mat['data']  # variable in mat file
    return mdata

##TP Definiciones
bandasRango = OrderedDict([("delta", (0,4.0)), ("theta", (4.0,8.0)), ("alpha", (8.0,13.0)), ("beta", (13.0,30.0)), ("gamma", (30.0,126.0))])
bandas = bandasRango.keys() 

##TP Definiciones
labels=['P','S']
pacientes=[]
for l in labels:
    pacientes += [l + "{:02d}".format(i) for i in range(1,  11)]


columnas = np.concatenate((
    ['name'],
    list(map(lambda n: n+'_m', bandas)), 
    list(map(lambda n: n+'_norm_m', bandas)), 
    list(map(lambda n: n+'_std', bandas)),
    list(map(lambda n: n+'_norm_std', bandas)),
    ['intra_media'],
    ['intra_desvio'],
    ['inter_media'],
    ['inter_desvio'],
    ['label']
))
columnas


##Definiciones Auxiliares
def bandaID(freq):
    for k, (_, v) in enumerate(bandasRango.items()):
        if freq >= v[0] and freq < v[1]:
            return k

# In[575]:

def load(name):
    """ Lee los datos desde filename (.mat) a un np array """    
    #sujeto_file = './DataSet/'+name+'.mat' #file path de un sujeto    
    #sujeto_file = '/media/laura/DISK_IMG/TP2/'+name+'.mat' #file path de un sujeto    
    sujeto_file = '/home/nico/Descargas/datos EEG/'+name+'.mat'
    return Mat2Data(sujeto_file)


# In[586]:

electrodos = [7, 43, 79, 130, 184]

# TODO consultar: TP2 b) Calcular los valores de cada banda de frecuencia, promediados entre los electrodos (todos) 
# y epochs para cada sujeto.
# Entiendo que debe ser las medias del analisis espectral en vez de el análisis espectral de las medias (b1)
# El cálculo del poder espectral en cada banda se realiza sumando los valores obtenidos de la FFT que corresponden a las
# frecuencias incluídas en la misma.
def poderEspectral(sujeto):
    poderEpochs = []
    poderNormalizado = []
    for epoch in range(0, len(sujeto)):#len(sujeto)
        poderElectrodos = []
        poderElectrodosNorm = []

        for electrodo in electrodos:
            freq, pot = analisis_espectral(sujeto[epoch][electrodo])                
            poderFreq = np.zeros(len(bandas))
            for f, p in zip(freq, pot):            
                poderFreq[bandaID(f)] += p
            
            poderElectrodos.append(poderFreq)
            suma= sum(poderFreq)
            poderElectrodosNorm.append(poderFreq/suma)
        
        poder=list(np.mean(poderElectrodos,axis=0))
        poderEpochs.append(poder)
        poderNorm = list(np.mean(poderElectrodosNorm, axis=0))
        poderNormalizado.append(poderNorm)
    return poderEpochs, poderNormalizado    

# Calcula las medidas de informacion de un paciente para
# utilizar en una como feature.
def informacionPorEpoch(sujeto):
    intraPorEpoch = []
    interPorEpoch = []

    for epoch in range(0, len(sujeto)):
        intraPorEpoch.append(entropia(sujeto[epoch][electrodos[0]]))

        serie1 = sujeto[epoch][electrodos[0]]
        serie2 = sujeto[epoch][electrodos[1]]
        interPorEpoch.append(entropia_conjunta(serie1, serie2))

    return intraPorEpoch, interPorEpoch


# In[587]:

def mainFeatures(outFile):
    data = pd.DataFrame({})
    for paciente in pacientes:
        print("Extrayendo features del paciente " + paciente)
        sujeto = load(paciente)
        poderEspectralEpochs, poderNormalizadoEpochs = poderEspectral(sujeto)
        
        # Espectrales
        media_espectral = np.mean(poderEspectralEpochs, axis=0)
        media_normalizados = np.mean(poderNormalizadoEpochs, axis=0)
        desvio_espectral = np.std(poderEspectralEpochs, axis=0)
        desvio_normalizado = np.std(poderNormalizadoEpochs, axis=0)

        # Informacion
        intraPorEpoch, interPorEpoch = informacionPorEpoch(sujeto)
        intra_media = np.mean(intraPorEpoch)
        intra_desvio = np.std(intraPorEpoch)
        inter_media = np.mean(interPorEpoch)
        inter_desvio = np.std(interPorEpoch)
        
        serie = pd.Series(np.concatenate(([paciente],media_espectral, 
                                                    media_normalizados, 
                                                    desvio_espectral, 
                                                    desvio_normalizado, 
                                                    [intra_media, intra_desvio, inter_media, inter_desvio],
                                                    [paciente[0]]), axis=0))
                         
        #save to csv
        df = pd.DataFrame({})
        df=df.append(serie, ignore_index=True)        
        df.to_csv('./features/'+outFile +'.csv', mode='a', header=False, index =False)        
        df.to_csv('./features/'+outFile + paciente + '.csv', mode='w', index =False)
        
        data = pd.concat([data, df], ignore_index=True)
    return data


# In[588]:
def generar():
    
    # Par generar el encabezado
    outFile = 'features'
    df = pd.DataFrame(columns=columnas)
    df.to_csv('./features/'+outFile+'.csv', columns=columnas, index =False)

    return mainFeatures(outFile)

# sin encabezados en el df, con encabezados en el cvs
"""df=generar()"""


# In[161]:


df = pd.read_csv('./features.csv')
df


# In[166]:

from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy import interp


# In[167]:

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

def featuresROC(df, ys):
    
    for k in range(1,25):
        name = columnas[k]
        print(name)
        xs = np.concatenate(df.loc[:,[name]].values)
        plotROC(xs, ys, name)



# In[171]:

#TODO consultar
#ys =['P']*10+['S']*10
"""
ys =['P']*10+['S']*10
featuresROC(df, ys)
"""

# In[172]:

def featuresCVROC(df, ys):
    
    for k in range(1,21):
        name = columnas[k]
        xs = np.concatenate(df.loc[:,[name]].values)
        plotCVROC(xs, ys, name)


# TODO consultar
def plotCVROC(xs, ys, name):
    y_true = label_binarize(ys, classes=['P', 'S']).ravel()
    
    # split X and y into training and testing sets    
    xs_train, xs_test, y_train, y_test = train_test_split(xs, y_true)
    
    #DeprecationWarning
    xs_train = xs_train.reshape(-1,1)
    xs_test = xs_test.reshape(-1,1)    
        
    # train a logistic regression model on the training set
    model = LogisticRegression()
    model = model.fit(xs_train, y_train)
    
    # make class predictions for the testing set
    #y_pred_class = model.predict(xs_test)    
    #y_score = model.predict(xs_test)
    y_predict_probabilities = model.predict_proba(xs_test)[:,0]#[:,1]
        
    # calculate accuracy
    #from sklearn import metrics
    #metrics.accuracy_score(y_test, y_pred_class)
    
    # Compute ROC curve and ROC area for each class     
    fpr, tpr, _ = roc_curve(y_test, y_predict_probabilities)
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



# In[174]:

#ys =['P']*10+['S']*10
"""ys =['P']*10+['S']*10

featuresCVROC(df, ys)
"""

# In[165]:
target_values = np.array(df.loc[range(2,18), ['label']]).ravel()
samples = df.loc[range(2,18),columnas[1:25]]
        
clf = svm.SVC()
clf.fit(samples, target_values)
print(clf.predict(df.loc[[0,1,18,19],columnas[1:25]]))

# In[ ]:




# In[ ]:



