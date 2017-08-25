import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn


data0 = np.loadtxt('tiempos.txt', skiprows=1)
data = np.loadtxt('tiemposCorregidos.txt', skiprows=1)

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
"""

w_sol_lluvia= stats.wilcoxon(data[:,1], data[:,3])
w_nublado_lluvia= stats.wilcoxon(data[:,2], data[:,3])
w_sol_nublado= stats.wilcoxon(data[:,1], data[:,2])
print("wilcoxon_tiempo_sol tiempo_lluvia:", w_sol_lluvia)
print("wilcoxon_tiempo_nublado tiempo_lluvia:", w_nublado_lluvia)
print("wilcoxon_tiempo_sol tiempo_nublado:", w_sol_nublado)

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