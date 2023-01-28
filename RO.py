import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import linprog
#Functions

def degres(M):
    D=[]
    for i in range(len(M)):
        deg=0
        for j in range(len(M[0])):
            if M[i][j]!=0 and M[i][j]!=np.inf:
                deg=deg+1
        D.append(deg)
    return D

def transform(M):
    new_M={}
    i = 0
    for x in M:
        x_dic = {}
        for y in x:
            if y != np.inf:
                x_dic[np.where(y==x)[0][0]] = y
        print(x_dic)
        new_M[i] = x_dic
        i += 1
    return new_M

def eulerien(M):
    D=degres(M)
    test=0
    est_eulerien = True
    for i in range(len(D)):
        if D[i]%2!=0:
            est_eulerien=False 
            test=test+1
    est_semi_eulirien = (test==2)
    if est_eulerien:
        return 'Ce graphe est `Eulerien`'
    elif est_semi_eulirien:
        return 'Ce graphe est `Semi Eulerien`'
    else:
        return 'Ce graphe est ni `Eulerien` ni `Semi Eulerien`'
        

def dijkstra(S, M):
    test = False
    for i in range(len(M)):
        for j in range(len(M)):
            if M[i][j] < 0:
                test = True
    if test:
        st.error('On peut pas appliquer dijksta sur un Graphe qui a des poids negatifs!')
    else:
        # Nombre de sommets du graphe
        n = len(M)

        # Initialisation de la distance et du predecesseur de chaque sommet
        distance = [np.inf for element in range(n)]
        predecesseur = [-1 for element in range(n)]
        predecesseur[S] = S
        distance[S] = 0

        # Initialisation de l'ensemble des sommets non encore inclus dans le plus court chemin
        non_visité = [element for element in range(n)]

        while non_visité:
            # Recherche du sommet non encore inclus avec la distance minimale
            min_distance = np.inf
            a_traiter = None
            for u in non_visité:
                if distance[u] < min_distance:
                    min_distance = distance[u]
                    a_traiter = u

            # Mise à jour de la distance et du predecesseur de chaque sommet voisin
            non_visité.remove(a_traiter)
            for v in range(n):
                if M[a_traiter][v] is not None:
                    n_distance = distance[a_traiter] + M[a_traiter][v]
                    if n_distance < distance[v]:
                        distance[v] = n_distance
                        predecesseur[v] = a_traiter
        test1 = False
        for i in range(len(M)):
            for j in range(len(M)):
                if distance[i] + M[i][j] < distance[j]:
                    test1 = True
        if test1:
            st.error('le graphe contient un circuit absorbant')
        else:
                        
            predecesseur_modifié = [names_list[x] for x in predecesseur]
            dis_pre_df = pd.DataFrame([distance, predecesseur_modifié], index=['distance','predecesseur'], columns=names_list)
            st.subheader('Votre tableau finale est de la forme:')
            st.dataframe(dis_pre_df, use_container_width=True)
            st.subheader('LES PCC:')
            for i in range(n):
                text = ''
                if i == S:
                    continue
                chemin = [i]
                j = predecesseur[i]
                while j != S:
                    chemin.append(j)
                    j = predecesseur[j]
                chemin.append(S)
                for i, x in enumerate(chemin[::-1]):
                    if i == len(chemin[::-1])-1:
                        text += f'`{names_list[x].upper()}`'
                    else:
                        text += f'`{names_list[x].upper()}` ➤'
                st.write(f"Le plus court chemin de {names_list[S]} vers {names_list[i]} est : {text}")
            # Affichage de l'arborescence des plus courts chemins
            tous_chemin = [] #liste qui va contenir les plus courts chemins vers tous les tous les sommets a partir du sommet S
            for i in range(n):
                if i == S:
                    continue
                chemin = [i]
                j = predecesseur[i]
                while j != S:
                    chemin.append(j)
                    j = predecesseur[j]
                chemin.append(S)
                tous_chemin.append(chemin[::-1])
            proches_couples = list() #liste qui va contenir juste les voisins du graphe
            for x in tous_chemin: 
                for i in range(len(x)-1):
                    if (names_list[x[i]],names_list[x[i+1]]) not in proches_couples:
                        proches_couples.append((names_list[x[i]], names_list[x[i+1]]))
            G = nx.DiGraph() #creation de notre graphe

            for el in proches_couples:
                G.add_edges_from([el], weight=round(M[names_list.index(el[0])][names_list.index(el[1])])) #creation des arretes (edges) qui lient les plus voisins,
                                                                    # en les attribuant le poids entre chaque couple.

            x=dict([((u,v,),d['weight'])
                            for u,v,d in G.edges(data=True)]) #les labels des arretes qui sont les poids dans ce cas
            fig,ax = plt.subplots(figsize = (6,3))
            fig.tight_layout()
            pos=nx.spring_layout(G) #creation des position de chaque sommet
            nx.draw_networkx_nodes(G, pos, node_color='blue') #colorier les sommets en bleu
            nx.draw_networkx_labels(G, pos,font_color='white') #coleur du texte au sein des sommets
            nx.draw_networkx_edges(G, pos, edge_color='navy', arrows = True,arrowsize=15) #ajustement des paramètres des arretes
            nx.draw_networkx_edge_labels(G, pos,edge_labels=x) # dessiner notre graphe
            st.subheader('Votre graphe est de la forme:')
            st.pyplot(fig)


class Graph:



    def __init__(self, vertices):

        self.M = vertices   # Total number of vertices in the graph

        self.graph = []     # Array of edges



    # Add edges

    def add_edge(self, a, b, c):

        self.graph.append([a, b, c])



    # Print the solution

    def print_solution(self, distance):

        st.subheader("distance des sommets à partir de la source:")

        for k in range(self.M):

            st.write(f'distance vers `{names_list[k]}`est: `{distance[k]}`')



    def bellman_ford(self, src):



        distance = [float("Inf")] * self.M

        distance[src] = 0



        for _ in range(self.M - 1):

            for a, b, c in self.graph:

                if distance[a] != float("Inf") and distance[a] + c < distance[b]:

                    distance[b] = distance[a] + c



        for a, b, c in self.graph:

            if distance[a] != float("Inf") and distance[a] + c < distance[b]:

                st.info("le graphe contient un cycle de longueur négative")
                return


        self.print_solution(distance)
        return




st.set_page_config(layout='wide')

fonctions = ["PCC","Simplexe"]

with st.sidebar:
    choice = st.selectbox("Qu'est ce que vous voulez faire?", fonctions)

st.markdown('<h2 style="text-align:center;">Recherche operationnelle</h2>', True)

with st.expander("A PROPOS DE L'APPLICATION"):
    st.write("Cette application vous aidera à obtenir les chemins les plus courts sur votre graphique ou à utiliser la méthode simplexe pour maximiser ou minimiser une fonction.")


if choice == 'PCC':
    st.markdown('<h4>🤗Saisie de la matrice :</h4>', True)
    M = np.array([])
    test = False
    sommets_num = st.number_input("Nombre de sommets du graphe:", min_value=0, max_value=20, value=0, step=1)
    if sommets_num > 1:
        names = st.text_input('Entrer le nom de vos sommets separés par des virgules :')
        names_list = names.strip().split(',')
        if sommets_num != len(names_list) and names !='' :
            st.error(f'vous devez saisir {sommets_num} sommets ')

        if sommets_num == len(names_list) :  
            st.markdown("<p ';>⛔Entrer la matrice d'adjacence en respectant l'ordre des noms que vous avez introduit! </p> <p>⛔Entrer <span style='color:green'>inf</span> dans les cases des sommets n'est pas liés </p> ", True)
            with st.form('my_form', False):  
                columns = st.columns(sommets_num)
                c = 0
                l = 0
                for x in range(sommets_num):
                    liste = []
                    with columns[x]:
                        for y in range(sommets_num):
                            if x == y:
                                if x == 0:
                                    st.text_input(names_list[y].upper(), key=str(x) + str(y), value=0)
                                else:
                                    st.text_input('', key=str(x) + str(y), value=0)                                    
                            else:
                                if y == 0:
                                    st.text_input(names_list[x].upper(), key=str(x) + str(y), value='inf')
                                elif x ==0 :
                                    st.text_input(names_list[y].upper(), key=str(x) + str(y), value='inf')                                    
                                else:
                                    st.text_input('', key=str(x) + str(y), value='inf')                                    
                        
                racine = st.selectbox('Choisir le sommet de départ:', names_list)
                algorithme = st.selectbox("Choisir l'algorithme", ['', 'Dijkstra','Bellman-Ford'])
                submitted = st.form_submit_button('envoyer')
            if submitted:
                for i in range(sommets_num):
                    for j in range(sommets_num):
                        if st.session_state[str(i) + str(j)] != 'inf':
                            M = np.append(M,int(st.session_state[str(i) + str(j)]) )
                        elif st.session_state[str(i) + str(j)] == 'inf':
                            M = np.append(M, np.inf)
                M = np.transpose(M.reshape((sommets_num, sommets_num)))
                matrix_df = pd.DataFrame(M, columns=names_list, index=names_list)
                st.subheader('Votre matrice est:')
                st.dataframe(matrix_df, use_container_width=True )
                st.subheader("Degré des sommets:")
                D = degres(M)
                df_degre = pd.DataFrame([D], index=["Degré"], columns = names_list)
                st.dataframe(df_degre, use_container_width=True)
                st.subheader('Nature du graphe:')
                st.write(eulerien(M))
                if algorithme == "Dijkstra":
                    try:
                        st.markdown('<h2 style="text-align:center;"> Algorithme de Dijkstra', True)
                        racine_int = names_list.index(racine)
                        dijkstra(racine_int, M)
                    except:
                        st.error('dijkstra ne peut pas etre appliqué a ce graphe')
                elif algorithme == 'Bellman-Ford':
                    try : 
                        st.markdown('<h2 style="text-align:center;"> Algorithme de Bellman-Ford', True)
                        racine_int = names_list.index(racine)
                        G = Graph(sommets_num)
                        for i in range(sommets_num):
                            for j in range(sommets_num):
                                if M[i][j] != np.inf:
                                    G.add_edge(i, j, M[i][j])
                        G.bellman_ford(racine_int)
                    except:
                        st.error('Bellman-Ford ne peut pas etre appliqué a ce graphe!')
elif choice  == "Simplexe":
    st.markdown("<h3 style='text-align:center'Simplexe> Methode du Simplexe", True)
    with st.form('my_form2', False):
        layout = st.columns(3)
        with layout[0]:
                    num_vars = st.number_input('Nombre des variables de decision.', step=1)
        with layout[1]:
                    num_cons = st.number_input('Nombre des contraintes.', step=1)
        with layout[2]:
            sense = st.selectbox('Objective de la Fonction', ['Maximiser', 'Minimiser'])
        if num_vars == 0:
            st.info('Veuillez entrer le nombre des variables!')
        if num_cons == 0:
            st.info('Veuillez entrer le nombre des contraintes!')
        submit = st.form_submit_button('Envoyer')
    with st.form('my_form3', False):
        st.markdown('<h4 style="text-align:center;"> La Fonction', True)
        if num_vars == 0 or num_cons == 0 :
            st.info('Veuillez remplir les cases ci-dessus:☝')
            submit2 = st.form_submit_button('Envoyer', disabled=True)
        else:
            st.markdown('<h5> Les Coefficients', True)
            coef = [0]*num_vars
            constraints = []
            ordres = []
            constantes = []
            columns = st.columns(num_vars)
            for i in range(num_vars):
                with columns[i]:
                    coef[i] = st.number_input(f'X{i+1}')
            st.markdown('<h5> Les Contraintes', True)
            columns2 = st.columns(num_vars + 2)
            for i in range(num_vars + 2):
                liste = []
                for j in range(num_cons):
                    if i < num_vars:
                        with columns2[i]:
                            constraint = st.number_input(f'X{i+1}', key = str(i) + str(j))
                            liste.append(constraint)
                    elif i == num_vars:
                        with columns2[i]:
                            ordre = st.selectbox('',['⩽','⩾' ], key = j)
                            ordres.append(ordre)
                    else:
                        with columns2[i]:
                            cste = st.number_input(f'C{j+1}')
                            constantes.append(cste)
                if i < num_vars:
                    constraints.append(liste)
            text = ''
            for i in range(num_vars):
                if i == num_vars -1:
                    text += f'X{i+1} ⩾0'
                else:
                    text += f'X{i+1},\t'
            st.markdown(f'<p style="text-align:center;"> {text}', True)
            constraints = np.transpose(constraints)
            coef = np.array(coef) * (-1)**(sense == 'Maximiser')
            constraints = [constraint * (-1)**(ordre == '⩾') for constraint, ordre in zip(constraints, ordres)]
            constantes = np.array(constantes)
            constantes = [constante * (-1)**(ordre == '⩾') for constante, ordre in zip(constantes, ordres)]
            for i in range(num_vars):
                liste = np.zeros(num_vars)
                liste[i] = -1
                constraints = np.append(constraints, [liste], axis=0)
                constantes= np.append(constantes, 0)


            submit2 = st.form_submit_button('Envoyer')
            if submit2:
                res = linprog(coef, A_ub=constraints, b_ub=constantes, method='simplex')
                st.markdown('<h3 style="text-align:center;margin-bottom:20px;">Results', True)
                col4 = st.columns([1,1,1])
                with col4[1]:
                    st.markdown(f'`➤` STAUT: `{res.message}`', True )
                    st.markdown(f"`➤` NOMBRE D'ITÉRATIONS:  `{res.nit}`", True)
                    st.write('`➤` VARIABLES DE DECISIONS:', pd.DataFrame([res.x], columns=[f'X{i+1}' for i in range(num_vars)], index=['values']))
                    opt = round(res.fun*((-1)**(sense != 'Minimiser')), ndigits=2)
                    st.markdown(f"`➤` L'OPTIMUM : `{opt }`")
st.sidebar.markdown('<p style="color:gray;font-size:0.8rem;font-family:monospace;text-align:center;"> Created with ❤️ by <a style="text-decoration:none;color:gray;" href="https://www.linkedin.com/in/achraf-moubarik-35086a24b/"> Moubarik Ashraf </a> </p>', True)
