import numpy as np
from sklearn import datasets, linear_model, svm, decomposition
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree.export import export_text
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori, association_rules
from mpl_toolkits.mplot3d import Axes3D
import pyfpgrowth

tienda = [['pan', 'leche', 'mantequilla', 'cerveza'],
          ['pan', 'mantequilla', 'agua', 'mermelada', 'cerveza'],
          ['cerveza', 'pañales', 'pan', 'mantequilla', 'mermelada'],
          ['mantequilla', 'leche', 'jugo'],
          ['pañales', 'cerveza', 'jugo', 'agua']]

ketchap = [['Leche', 'Cebolla', 'Nuez Moscada', 'Frijoles', 'Huevos', 'Yogurt'],
           ['Cilantro', 'Cebolla', 'Nuez Moscada', 'Frijoles', 'Huevos', 'Yogurt'],
           ['Leche', 'Manzana', 'Frijoles', 'Huevos'],
           ['Leche', 'Maiz dulce', 'Maiz', 'Frijoles', 'Yogurt'],
           ['Maiz', 'Cebolla', 'Cebolla', 'Frijoles', 'Helado', 'Huevos']]

def pickdataset(id):
    dataset = None
    if id == 1:
        dataset = datasets.load_wine()
    elif id == 2:
        dataset = datasets.load_breast_cancer()
    elif id == 3:
        dataset = datasets.load_iris()
    elif id == 4:
        dataset = datasets.load_diabetes()
    elif id == 5:
        dataset = datasets.load_digits()
    elif id == 6:
        dataset = tienda
    elif id == 7:
        dataset = ketchap
    return dataset


def bayesA(id, principal):
    gnb = GaussianNB()
    dataset = pickdataset(int(id))
    pasos = "DataSet cargado: iris" + '\n'
    y_pred = gnb.fit(dataset.data, dataset.target).predict(dataset.data)
    matrizPorcentaje = metrics.classification_report(dataset.target, y_pred)
    confusion = metrics.confusion_matrix(dataset.target, y_pred)
    avgReal = str(metrics.accuracy_score(dataset.target, y_pred) * 100) + '%'
    pasos += "Target: " + '\n' + str(dataset.target) + '\n'
    pasos += "Resultados: " + '\n' + str(y_pred) + '\n'
    pasos += "Matriz de porcentajes: " + '\n' + matrizPorcentaje + '\n'
    pasos += "Matriz de confución: " + '\n' + str(confusion) + '\n'
    if principal:
        context = {'algoritmoPrincipal': 'Naive Bayes', 'resultado': avgReal, 'pasos': pasos, 'reglas': 'No aplica'}
    else:
        context = {'algoritmoComparar': 'Naive Bayes', 'resultado2': avgReal, 'pasos2': pasos, 'reglas2': 'No aplica', 'img2': 'No aplica'}
    return context


def guardarImagenKnn(dataset):
    X = dataset.data[:, :2]
    h = .02
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    knn = KNeighborsClassifier()
    knn.fit(X, dataset.target)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, -1], c=dataset.target, cmap=cmap_bold)
    plt.title('Resultado KNN')
    plt.savefig('static/img/knn.png')
    #plt.close()


def guardarImagenKMeans(kmeans, dataset):
    X = dataset.data
    plt.figure()
    plt.scatter(X[:, 0], X[:, -1], c=kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, -1], color='red', marker='o')
    plt.title("Kmeans 4 clusters")
    plt.savefig('static/img/kmeans.png')
    #plt.close()


def guardarImagenRegresion(dataset, datos, xTest, y_pred):

    plt.figure()
    plt.scatter(datos, dataset.target, color='black')
    plt.plot(xTest, y_pred, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.title("Grafica de la linea")
    plt.savefig('static/img/regresion.png')


def guardarImagenId3(id3):
    plt.figure()
    plot_tree(id3, filled=True)
    plt.title("Arbol ID3")
    plt.savefig('static/img/id3.png')


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def guardarImagenSVC(X, modelo, target):
    plt.figure()
    modelo = modelo.fit(X, target)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z)
    plt.scatter(X0, X1, c=target, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.title('Maquinas de sooprte con Kernel')
    plt.savefig('static/img/svm.png')


def guardarImagenPCA(datos, targets):
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
    for label in targets:
        ax.text3D(datos[targets == label, 0].mean(),
                  datos[targets == label, 1].mean() + 1.5,
                  datos[targets == label, 2].mean(), label,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(targets, [1, 2, 0]).astype(np.float)
    ax.scatter(datos[:, 0], datos[:, 1], datos[:, 2], c=y, cmap=plt.cm.nipy_spectral,
               edgecolor='k')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.title('PCA con 3 caracteristicas')
    plt.savefig('static/img/pca.png')


def knn(id, principal):
    pasos = "Dataset cargado" + '\n'
    dataset = pickdataset(int(id))
    knn = KNeighborsClassifier()
    knn.fit(dataset.data, dataset.target)
    y_pred = knn.predict(dataset.data)
    guardarImagenKnn(dataset)
    matrizPorcentaje = metrics.classification_report(dataset.target, y_pred)
    confusion = metrics.confusion_matrix(dataset.target, y_pred)
    grafo = knn.kneighbors_graph(dataset.data)
    avgReal = str(metrics.accuracy_score(dataset.target, y_pred) * 100) + '%'
    pasos += "Target: " + '\n' + str(dataset.target) + '\n'
    pasos += "Resultados: " + '\n' + str(y_pred) + '\n'
    pasos += "Matriz de porcentajes: " + '\n' + matrizPorcentaje + '\n'
    pasos += "Matriz de confución: " + '\n' + str(confusion) + '\n'
    reglas = str(grafo.toarray()) + '\n'
    guardarImagenKnn(dataset)
    img = 'knn.png'
    if principal:
        context = {'algoritmoPrincipal': 'K-NN', 'resultado': avgReal, 'pasos': pasos, 'reglas': reglas, 'img': img}
    else:
        context = {'algoritmoComparar': 'K-NN', 'resultado2': avgReal, 'pasos2': pasos, 'reglas2': reglas, 'img2': img}
    return context


def kmeans(id, principal):
    pasos = "Dataset cargado" + '\n'
    dataset = pickdataset(int(id))
    kmeans = KMeans(n_clusters=4, max_iter=3000, algorithm='auto', random_state=0)
    kmeans.fit(dataset.data)
    y_pred = kmeans.predict(dataset.data)
    centroides = kmeans.cluster_centers_
    labels = kmeans.labels_
    avgReal = str(silhouette_score(dataset.data, kmeans.labels_) * 100) + '%'
    pasos += "Target: " + '\n' + str(dataset.target) + '\n'
    pasos += "Clusters: " + '\n' + str(kmeans.n_clusters) + '\n'
    pasos += "Resultados: " + '\n' + str(labels) + '\n'
    pasos += "Centroides: " + '\n' + str(centroides) + '\n'
    pasos += "Iteraciones: " + '\n' + str(kmeans.n_iter_) + '\n'
    guardarImagenKMeans(kmeans, dataset)
    img = 'kmeans.png'
    if principal:
        context = {'algoritmoPrincipal': 'KMeans', 'resultado': avgReal, 'pasos': pasos, 'reglas': "Centroides:" + str(centroides), 'img': img}
    else:
        context = {'algoritmoComparar': 'KMeans', 'resultado2': avgReal, 'pasos2': pasos, 'reglas2': "Centroides:" + str(centroides), 'img2': img}
    return context


def id3(id, principal):
    pasos = "Dataset cargado" + '\n'
    dataset = pickdataset(int(id))
    decision_tree = DecisionTreeClassifier(random_state=0)
    decision_tree = decision_tree.fit(dataset.data, dataset.target)
    guardarImagenId3(decision_tree)

    arbol = export_text(decision_tree, feature_names=dataset['feature_names'])
    reglas = arbol
    puntaje = cross_val_score(decision_tree, dataset.data, dataset.target, cv=10)
    pasos += "Peso de los nodos: " + str(decision_tree.tree_.weighted_n_node_samples) + '\n'
    pasos += "Impureza: " + str(decision_tree.tree_.impurity) + '\n'
    pasos += "Altura del arbol: " + str(decision_tree.get_depth()+1) + '\n'
    pasos += "Hojas: " + str(decision_tree.get_n_leaves()) + '\n'
    pasos += "Puntaje: " + str(puntaje) + '\n'
    img = 'id3.png'
    avgReal = str(decision_tree.score(dataset.data, dataset.target) * 100)
    if principal:
        context = {'algoritmoPrincipal': 'ID3', 'resultado': avgReal, 'pasos': pasos,
                   'reglas': reglas, 'img': img}
    else:
        context = {'algoritmoComparar': 'ID3', 'resultado2': avgReal, 'pasos2': pasos,
                   'reglas2': reglas, 'img2': img}
    return context


def regresionLineal(id, principal):
    pasos = "Dataset cargado" + '\n'
    dataset = pickdataset(int(id))
    datos = dataset.data[:, np.newaxis, 2]
    xTrain =datos[:-20]
    xTest = datos[-20:]
    yTrain = dataset.target[:-20]
    yTest = dataset.target[-20:]
    regr = linear_model.LinearRegression()
    regr.fit(xTrain, yTrain)
    y_pred = regr.predict(xTest)
    guardarImagenRegresion(dataset, datos, xTest, y_pred)
    error = str(mean_squared_error(yTest, y_pred))
    varianza = r2_score(yTest, y_pred)
    pasos += "Coeficientes: " + str(regr.coef_) + '\n'
    pasos += "Pendiente: " + str(regr.coef_.T) + '\n'
    pasos += "Intercepto: " + str(regr.intercept_) + '\n'
    pasos += "Error cuadrado:" + error + '\n'
    pasos += "Varianza: " + str(varianza) + '\n'
    avgReal = r2_score(yTest, y_pred) * 100
    reglas = "Y = " + str(regr.coef_.T) + "x +" + str(regr.intercept_)
    img = 'regresion.png'
    if principal:
        context = {'algoritmoPrincipal': 'Regresión Lineal', 'resultado': avgReal, 'pasos': pasos,
                   'reglas': reglas, 'img': img}
    else:
        context = {'algoritmoComparar': 'Regresión Lineal', 'resultado2': avgReal, 'pasos2': pasos,
                   'reglas2': reglas, 'img2': img}
    return context


def Apriori(id, principal):
    pasos = "Dataset cargado" + '\n'
    dataset = pickdataset(int(id))
    oht = OnehotTransactions()
    oht_ary = oht.fit(dataset).transform(dataset)
    df = pd.DataFrame(oht_ary, columns=oht.columns_)
    frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
    association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    pasos += "Dataset Procesado: " + '\n'
    pasos += str(df) + '\n'
    pasos += "Item Set: " + '\n'
    pasos += str(frequent_itemsets) + '\n'
    avgReal = str(np.mean(rules.as_matrix(columns=['support'])) * 100) + "% soporte promedio"
    reglas = rules[['antecedents', 'consequents', 'support']]
    if principal:
        context = {'algoritmoPrincipal': 'Apriori', 'resultado': avgReal, 'pasos': pasos,
                   'reglas': reglas, 'img': 'No aplica'}
    else:
        context = {'algoritmoComparar': 'Regresión Lineal', 'resultado2': avgReal, 'pasos2': pasos,
                   'reglas2': reglas, 'img2': 'No aplica'}
    return context


def markov(id, principal):
    pasos = "Datos cargados " + '\n'
    if id == 8:
        states = ['Dormir', 'Comer', 'Trabajar']
        probaInic = [0.35, 0.35, 0.3]
    else:
        states = ['Verbo', 'Sustantivo', 'Adjetivo', 'Determinante']
        probaInic = [0.35, 0.30, 0.20, 0.15]
    state_space = pd.Series(probaInic, index=states, name='states')
    dataset = pd.DataFrame(columns=states, index=states)
    for i in range(0, len(states)):
        array = np.random.random(len(states))
        array = array / np.sum(array)
        dataset.loc[states[i]] = array
    edges = {}
    for col in dataset.columns:
        for idx in dataset.index:
            edges[(idx, col)] = dataset.loc[idx, col]
    pasos += "Estados: " + '\n' + str(state_space) + '\n'
    pasos += "Transiciones: " + '\n' + str(edges) + '\n'
    reglas = str(dataset)
    if principal:
        context = {'algoritmoPrincipal': 'Modelos Ocultos de Markov', 'resultado': 'No aplica', 'pasos': pasos,
                   'reglas': reglas, 'img': 'No aplica'}
    else:
        context = {'algoritmoComparar': 'Regresión Lineal', 'resultado2': 'No aplica', 'pasos2': pasos,
                   'reglas2': reglas, 'img2': 'No aplica'}
    return context


def Svm(id, principal):
    pasos = "Dataset Cargado" + '\n'
    dataset = pickdataset(int(id))
    datosX = dataset.data[:, :2]
    modelo = svm.SVC(kernel='rbf', gamma=0.7, C=1)
    entrenar = modelo.fit(dataset.data, dataset.target)
    guardarImagenSVC(datosX, modelo, dataset.target)
    pasos += "Vectores de soporte: " + '\n'
    pasos += str(modelo.support_vectors_) + '\n'
    pasos += "Soporte:" + '\n'
    pasos += str(modelo.support_) + '\n'
    reglas = "Funcion de desición:" + '\n'
    reglas += str(entrenar.decision_function(datosX)) + '\n'
    avgReal = str(entrenar.score(datosX, dataset.target) * 100) + '%'
    img = "svm.png"
    if principal:
        context = {'algoritmoPrincipal': 'Maquinas de Soporte Vectorial', 'resultado': avgReal, 'pasos': pasos,
                   'reglas': reglas, 'img': img}
    else:
        context = {'algoritmoComparar': 'Maquinas de Soporte Vectorial', 'resultado2': avgReal, 'pasos2': pasos,
                   'reglas2': reglas, 'img2': img}
    return context


def pca(id, principal):
    pasos = "Datos cargados" + '\n'
    dataset = pickdataset(int(id))
    modelo = decomposition.PCA(n_components=3)
    modelo.fit(dataset.data)
    guardarImagenPCA(dataset.data, dataset.target)
    pasos += "Componentes: " + '\n' + str(modelo.n_components) + '\n'
    pasos += "Componentes: " + '\n' + str(modelo.components_) + '\n'
    pasos += "Covarianza: " + '\n' + str(modelo.get_covariance()) + '\n'
    pasos += "Varianza explicada: " + '\n' + str(modelo.explained_variance_ ) + '\n'
    pasos += "Presicion: " + '\n' + str(modelo.get_precision()) + '\n'
    avgReal = str(modelo.score(dataset.data, dataset.target)) + '#'
    reglas = "Precisión:" + '\n'
    reglas += str(modelo.get_precision()) + '\n'
    img = 'pca.png'
    if principal:
        context = {'algoritmoPrincipal': 'PCA', 'resultado': avgReal, 'pasos': pasos,
                   'reglas': reglas, 'img': img}
    else:
        context = {'algoritmoComparar': 'PCA', 'resultado2': avgReal, 'pasos2': pasos,
                   'reglas2': reglas, 'img2': img}
    return context


def fpgrouth(id, principal):
    pasos = "Dataset Cargado" + '\n'
    dataset = pickdataset(int(id))
    patterns = pyfpgrowth.find_frequent_patterns(dataset, 3)
    rules = pyfpgrowth.generate_association_rules(patterns, 0.6)
    pasos += "Encuentros: " + '\n'
    pasos += str(patterns) + '\n'
    avgReal = 0
    for i in rules.values():
        it = i[1:2]
        x = str(it)
        x1 = x.split(',')
        x2 = str(x1[0])
        x3 = x2.split('(')
        avgReal += float(x3[1])
    avgReal = str((avgReal / len(rules.values())) * 100) + '% Confianza'
    reglas = str(rules)
    img = 'No aplica'
    if principal:
        context = {'algoritmoPrincipal': 'FP-growth', 'resultado': avgReal, 'pasos': pasos,
                   'reglas': reglas, 'img': img}
    else:
        context = {'algoritmoComparar': 'FP-growth', 'resultado2': avgReal, 'pasos2': pasos,
                   'reglas2': reglas, 'img2': img}
    return context

