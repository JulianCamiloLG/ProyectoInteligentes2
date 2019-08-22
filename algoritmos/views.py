from django.shortcuts import render
from .algoritmos import *
import csv
import codecs
import pandas as pd

# Create your views here.
def home(request):
    return render(request, 'index.html')


def elecciones(request):
    return render(request, 'elecciones.html')


def leerDataset(dataset):
    datos = pd.read_csv(dataset)

    #labels = list(datos.columns.values)
    #Y = datos.pop(labels[len(labels)-1])
    #print(Y)
    #print("-------------------------------------------------------------------")

    #X = datos.loc[:, datos.keys()]
    #datitos = X[:, np.newaxis, 2]

    return datos


def resultados(request):
    if request.method == 'POST':
        algoritmoP =request.POST['algoritmoPrincipal']
        algoritmoC = request.POST['algoritmoComparar']
        idDataset = request.POST['dataset']
        dataset = {}
        print(idDataset)
        if idDataset == '0':
            datos = request.FILES['datasetfile']
            print(datos)
            dataset = leerDataset(datos)
        context = {}
        if algoritmoP == 'Bayes':
            context = bayesA(idDataset, dataset, True)
        elif algoritmoP == 'Knn':
            context = knn(idDataset, dataset, True)
        elif algoritmoP == 'Kmeans':
            context = kmeans(idDataset, dataset, True)
        elif algoritmoP == 'id3':
            context = id3(idDataset, dataset, True)
        elif algoritmoP == 'regresion':
            context = regresionLineal(idDataset, dataset, True)
        elif algoritmoP == 'apriori':
            context = Apriori(idDataset, dataset, True)
        elif algoritmoP == 'hmm':
            context = markov(idDataset, dataset, True)
        elif algoritmoP == 'svm':
            context = Svm(idDataset, dataset, True)
        elif algoritmoP == 'pca':
            context = pca(idDataset, dataset, True)
        elif algoritmoP == 'fpgrouth':
            context = fpgrouth(idDataset, dataset, True)

        if algoritmoC == 'Bayes':
            context2 = bayesA(idDataset, dataset, False)
            context.update(context2)
        elif algoritmoC == 'Knn':
            context2 = knn(idDataset, dataset, False)
            context.update(context2)
        elif algoritmoC == 'Kmeans':
            context2 = kmeans(idDataset, dataset, False)
            context.update(context2)
        elif algoritmoC == 'regresion':
            context2 = regresionLineal(idDataset, dataset, False)
            context.update(context2)
        elif algoritmoC == 'apriori':
            context2 = Apriori(idDataset, dataset, False)
            context.update(context2)
        elif algoritmoC == 'hmm':
            context2 = markov(idDataset, dataset, False)
            context.update(context2)
        elif algoritmoC == 'svm':
            context2 = Svm(idDataset, dataset, False)
            context.update(context2)
        elif algoritmoC == 'pca':
            context2 = pca(idDataset, dataset, False)
            context.update(context2)
        elif algoritmoC == 'fpgrouth':
            context2 = fpgrouth(idDataset, dataset, False)
            context.update(context2)

        return render(request, 'resultados.html', context)
    else:
        return render(request, 'resultados.html')


def neuronal(request):
    return render(request, 'neuronal.html')


