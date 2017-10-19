import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import csv
import networkx as nx
import seaborn
from random import choice

def get_autores(linea):
    autores = linea[3]
    car = '\/ .~-()`'
    for c in car:
        autores = autores.replace(c, '')
    return autores.upper().split('&')

def add_paper(G, linea):
    autores = get_autores(linea)
    
    for autor in autores:
        G.add_node(autor)
    
    e = 1
    for autor in autores:
        if e < len(autores):
            edges = zip([autor]*(len(autores)-e), autores[e:])
            G.add_edges_from(edges)
        e = e + 1

# Punto 1: Creacion del grafo
with open('data.csv', 'r') as data:
    papers = csv.reader(data)
    G = nx.Graph()
    for paper in papers:
        add_paper(G, paper)

cant_nodos = len(list(G.nodes()))
cant_aristas = len(list(G.edges()))
print("Cantidad de nodos: " + str(cant_nodos))
print("Cantidad de aristas: " + str(cant_aristas))

# Punto 2: Distribucion de grados
dg = sorted([e[1] for e in list(nx.degree(G))], reverse=True)  

plt.plot(dg)
plt.title('Distribución de grado (lineal)')
plt.xlabel('Nodo')
plt.ylabel('Grado')
plt.savefig('graficos/dg_lineal.jpg', dpi=300)
plt.close()

plt.loglog(dg)
plt.title('Distribución de grado (loglog)')
plt.xlabel('Nodo')
plt.ylabel('Grado')
plt.savefig('graficos/dg_loglog.jpg', dpi=300)
plt.close()

plt.semilogx(dg)
plt.title('Distribución de grado (semilog x)')
plt.xlabel('Nodo')
plt.ylabel('Grado')
plt.savefig('graficos/dg_semilogx.jpg', dpi=300)
plt.close()

# Punto 3 : Componentes Conexas
ncc = nx.number_connected_components(G)
print("Numero de componentes conexas: " + str(ncc))
    
    # cg es la componente gigante
cg = max(nx.connected_component_subgraphs(G), key=len)
tamcg = len(list(cg.nodes()))
print("Tamaño de la componente gigante: " + str(tamcg))

# Punto 4 : Componentes Conexas
def tamañoVecino(cg):
    a=[]
    c=[]
    random_node = choice(list(cg.nodes()))
    for i in range (0,20):
        b=[]
        b=nx.ego_graph(cg, random_node, radius=i, center=False)
        if i==0:
            c.append(len(b))
        else:   
            c.append(len(b)-(a[i-1]))
 
        a.append(len(b))
    return a , c , c.index(max(c))



iterations=np.arange(1,10,1)
n= len(iterations)
colors = mpl.cm.gist_rainbow(np.linspace(0, 1, n))
maximos=[]

fig, ax = plt.subplots()
for color, i in zip(colors, iterations):
    ax.plot(tamañoVecino(cg)[0],color=color)
plt.title('Tamaño de vecindades 1')
plt.xlabel('Paso/distancia')
plt.ylabel('Numero de autores alcanzados')
plt.savefig('graficos/tamañodeVecindades1.jpg', dpi=300)
plt.close()

fig, ax1 = plt.subplots()
for color, i in zip(colors, iterations):
    ax1.plot(tamañoVecino(cg)[1],color=color)
    maximos.append(tamañoVecino(cg)[2])
plt.title('Tamaño de vecindades 2')
plt.xlabel('Paso')
plt.ylabel('Numero de autores nuevos')
plt.savefig('graficos/tamañodeVecindades2.jpg', dpi=300)
plt.close()

print(np.mean(maximos))
    

    
    
    
    
    


