import numpy as np
from sklearn import datasets, linear_model
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
    #plt.close()


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


