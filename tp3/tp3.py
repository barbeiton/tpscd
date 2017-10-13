
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
    sujeto_file = './DataSet/'+name+'.mat' #file path de un sujeto    
    #sujeto_file = '/media/laura/DISK_IMG/TP2/'+name+'.mat' #file path de un sujeto    
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

# In[587]:

def mainFeatures(outFile):
    data = pd.DataFrame({})
    for paciente in pacientes:
        sujeto = load(paciente)
        poderEspectralEpochs, poderNormalizadoEpochs = poderEspectral(sujeto)
        
        media = np.mean(poderEspectralEpochs, axis=0)
        media_normalizados = np.mean(poderNormalizadoEpochs, axis=0)
        desvio = np.std(poderEspectralEpochs, axis=0)
        desvio_normalizado = np.std(poderNormalizadoEpochs, axis=0)
        
        serie = pd.Series(np.concatenate(([paciente],media, media_normalizados, desvio, desvio_normalizado, [paciente[0]]), axis=0))
                         
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
#df=generar()


# In[161]:


df = pd.read_csv('./features/features.csv')
df


# In[166]:

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
    
    for k in range(1,21):
        name = columnas[k]
        print(name)
        xs = np.concatenate(df.loc[:,[name]].values)
        plotROC(xs, ys, name)



# In[171]:

#TODO consultar
#ys =['P']*10+['S']*10
ys =['P']*10+['S']*10
featuresROC(df, ys)


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
ys =['P']*10+['S']*10

featuresCVROC(df, ys)




# In[165]:

# Una medida de información intra-electrodo (a elección)

# Una medida de informacion inter-electrodo (a elección)


# In[ ]:




# In[ ]:



