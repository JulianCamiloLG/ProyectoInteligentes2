from django.shortcuts import render
from .algoritmos import *


# Create your views here.
def home(request):
    return render(request, 'index.html')


def elecciones(request):
    return render(request, 'elecciones.html')


def resultados(request):
    if request.method == 'POST':
        algoritmoP = request.POST['algoritmoPrincipal']
        algoritmoC = request.POST['algoritmoComparar']
        idDataset = request.POST['dataset']
        context = {}
        if algoritmoP == 'Bayes':
            context = bayesA(idDataset, True)
        elif algoritmoP == 'Knn':
            context = knn(idDataset, True)
        elif algoritmoP == 'Kmeans':
            context = kmeans(idDataset, True)
        elif algoritmoP == 'id3':
            context = id3(idDataset, True)
        elif algoritmoP == 'regresion':
            context = regresionLineal(idDataset, True)
        elif algoritmoP == 'apriori':
            context = Apriori(idDataset, True)

        if algoritmoC == 'Bayes':
            context2 = bayesA(idDataset, False)
            context.update(context2)
        elif algoritmoC == 'Knn':
            context2 = knn(idDataset, False)
            context.update(context2)
        elif algoritmoC == 'Kmeans':
            context2 = kmeans(idDataset, False)
            context.update(context2)
        elif algoritmoC == 'regresion':
            context2 = regresionLineal(idDataset, False)
            context.update(context2)
        elif algoritmoC == 'apriori':
            context2 = Apriori(idDataset, False)
            context.update(context2)

        return render(request, 'resultados.html', context)
    else:
        return render(request, 'resultados.html')


def neuronal(request):
    return render(request, 'neuronal.html')


