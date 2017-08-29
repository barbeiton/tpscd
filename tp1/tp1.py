import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn
import random


data0 = np.loadtxt('tiempos.txt', skiprows=1)
data = np.loadtxt('tiemposCorregidos.txt', skiprows=1)



# Test de permutacio'n para la media de los tiempos entre 
# di'as soleados y lluviosos


tiempos_sol = data[:,1] 

tiempos_llu = data[:,3]

delta0 = np.mean(tiempos_llu) - np.mean(tiempos_sol)

times = 1000
deltas = []
for i in range(0, times):
    tsol = 0
    tllu = 0
    for tiempo in zip(tiempos_sol, tiempos_llu):
        r = random.random()
        if  r < 0.5:
            tsol += tiempo[0]
            tllu += tiempo[1]
        else:
            tsol += tiempo[1]
            tllu += tiempo[0]


    deltas.insert(0, tllu/len(tiempos_llu) - tsol/len(tiempos_sol))

plt.hist(deltas, facecolor='g', alpha=0.75)
plt.grid(True)
plt.axvline(delta0, color='r')
plt.xlabel('Deltas')
plt.show()


"""
plt.figure()
plt.scatter(data0[:,0], data0[:,1],label = "Soleado")
plt.scatter(data0[:,0], data0[:,2],label = "Nublado")
plt.scatter(data0[:,0], data0[:,3],label = "Lluvioso")
plt.legend()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),fancybox=True, shadow=True, ncol=6)
plt.xlabel('Atleta')
plt.ylabel('Tiempos ')
plt.xlim(0,13)
plt.savefig("Datos0", dpi = 300)
plt.close()   # Terminar un gráfico



plt.figure()
plt.scatter(data[:,0], data[:,1],label = "Soleado")
plt.scatter(data[:,0], data[:,2],label = "Nublado")
plt.scatter(data[:,0], data[:,3],label = "Lluvioso")
plt.legend()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),fancybox=True, shadow=True, ncol=6)
plt.xlabel('Atleta')
plt.ylabel('Tiempos ')
plt.xlim(0,13)
plt.savefig("Datos", dpi = 300)
plt.close()   # Terminar un gráfico


n_tiempo_sol=stats.shapiro(data[:,1])
n_tiempo_nublado=stats.shapiro(data[:,2])
n_tiempo_lluvia=stats.shapiro(data[:,3])

print("normalidad tiempo_sol:", n_tiempo_sol)
print("normalidad tiempo_nublado:", n_tiempo_nublado)
print("normalidad tiempo_lluvia:", n_tiempo_lluvia)


w_sol_lluvia= stats.wilcoxon(data[:,1], data[:,3])
w_nublado_lluvia= stats.wilcoxon(data[:,2], data[:,3])
w_sol_nublado= stats.wilcoxon(data[:,1], data[:,2])
print("wilcoxon_tiempo_sol tiempo_lluvia:", w_sol_lluvia)
print("wilcoxon_tiempo_nublado tiempo_lluvia:", w_nublado_lluvia)
print("wilcoxon_tiempo_sol tiempo_nublado:", w_sol_nublado)
"""

"""
pearson_sol_nublado=stats.pearsonr(data[:,1], data[:,2])
print("tiempo_sol tempo_nublado:", pearson_sol_nublado)

pearson_sol_lluvia=stats.pearsonr(data[:,1], data[:,3])
print("tiempo_sol tiempo_lluvia:", pearson_sol_lluvia)

x3=stats.pearsonr(data[:,2], data[:,3])
print("tiempo_nublado tiempo_lluvia:", x3)



plt.figure()
plt.hist(data[:,1])
plt.title("Histograma día soleado")
plt.savefig("Hist_Soleado", dpi = 300)
plt.close()   # Terminar un gráfico

plt.figure()
plt.hist(data[:,2])
plt.title("Histograma día nublado")
plt.savefig("Hist_Nublado", dpi = 300)
plt.close()   # Terminar un gráfico

plt.figure()
plt.hist(data[:,3])
plt.title("Histograma día lluvioso")
plt.savefig("Hist_Lluvioso", dpi = 300)
plt.close()   # Terminar un gráfico

"""
