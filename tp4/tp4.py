import matplotlib.pyplot as plt
import csv
import networkx as nx

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


with open('data.csv', 'r') as data:
    papers = csv.reader(data)
    G = nx.Graph()
    for paper in papers:      
        add_paper(G, paper)
    
    print("Cantidad de nodos: " + str(len(list(G.nodes()))))
    print("Cantidad de aristas: " + str(len(list(G.edges()))))

    # Histograma de grados
    print(nx.degree(G))
    plt.hist(sorted([e[1] for e in list(nx.degree(G)) ], reverse=True))
    plt.show()
    

