import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import csv
import networkx as nx
import seaborn
from random import choice
import operator


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



# Punto 4 : Tamaños de Vecindades
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
plt.title('Tamaño de vecindades')
plt.xlabel('Paso/distancia')
plt.ylabel('Numero de autores alcanzados')
plt.savefig('graficos/tamañodeVecindades1.jpg', dpi=300)
plt.close()

fig, ax1 = plt.subplots()
for color, i in zip(colors, iterations):
    ax1.plot(tamañoVecino(cg)[1],color=color)
    maximos.append(tamañoVecino(cg)[2])
plt.title('Autores nuevos por paso')
plt.xlabel('Paso')
plt.ylabel('Numero de autores nuevos')
plt.savefig('graficos/tamañodeVecindades2.jpg', dpi=300)
plt.close()

print('Paso asociado al mayor número de autores nuevos - valor medio de todas las iteraciones: ' + str(np.mean(maximos)))
 



# Punto 5 : Mundos Pequeños
nx.draw(cg)  # networkx draw()
plt.draw()  # pyplot draw()
plt.show()


C= nx.average_clustering(cg)
print("Coeficiente de clustering C para cg: " + str(C))

l=nx.average_shortest_path_length(cg)
print("Camino mínimo medio l para cg: " + str(l))   

#random graph con la misma distribucón de grado    
rg = nx.random_degree_sequence_graph(sorted([e[1] for e in list(nx.degree(cg))], reverse=True))     

nx.draw(rg)  # networkx draw()
plt.draw()  # pyplot draw()
plt.show()

print("Coeficiente de clustering C para rg: " + str(nx.average_clustering(rg)))    
print("Camino mínimo medio l para rg: " + str(nx.average_shortest_path_length(rg)))
 






# Punto 6 : Estrellas
degree_dict = dict(cg.degree(cg.nodes())) # Run degree centrality
betweenness_dict = nx.betweenness_centrality(cg) # Run betweenness centrality
eigenvector_dict = nx.eigenvector_centrality(cg) # Run eigenvector centrality

# Assign each to an attribute in your network
nx.set_node_attributes(G, betweenness_dict, 'betweenness')
nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')
nx.set_node_attributes(cg, degree_dict, 'degree')

sorted_degree = sorted(degree_dict.items(), key=operator.itemgetter(1), reverse=True)
sorted_betweenness = sorted(betweenness_dict.items(), key=operator.itemgetter(1), reverse=True)
sorted_eigenvector = sorted(eigenvector_dict.items(), key=operator.itemgetter(1), reverse=True)

#top 20 nodes by degree as a list
top_degree = sorted_degree[:20]
print("Top 20 nodos por Degree")
for td in top_degree:
    print("Name:", td[0], "| Degree Centrality:", td[1])
print(" ")


#top 20 nodes by betweenness as a list
top_betweenness = sorted_betweenness[:20]


print("Top 20 nodos por Betweenness")
for tb in top_betweenness: # Loop through top_betweenness
    degree = degree_dict[tb[0]] # Use degree_dict to access a node's degree, see footnote 2
    eigenvector=eigenvector_dict[tb[0]]
    print("Name:", tb[0], "| Betweenness Centrality:", tb[1], "| Degree:", degree, "| Eigenvector:", eigenvector)

print(" ")

#top 20 nodes by eigenvector as a list
top_eigenvector = sorted_eigenvector[:20]
print("Top 20 nodos por Eigenvector")
for te in top_eigenvector: 
    degree = degree_dict[te[0]] 
    betweenness=betweenness_dict[te[0]]
    print("Name:", te[0], "| Eigenvector Centrality:", te[1], "| Degree:", degree, "| Betweenness:", eigenvector)

