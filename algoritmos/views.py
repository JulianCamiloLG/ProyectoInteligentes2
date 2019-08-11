from django.shortcuts import render, redirect, render_to_response
from django.views.generic import CreateView,UpdateView,ListView,DeleteView
from django.urls import reverse_lazy


# Create your views here.
def home(request):
    return render(request, 'index.html')


def elecciones(request):
    return render(request, 'elecciones.html')


def resultados(request):
    if request.method == 'POST':
        print(request.POST)
        algoritmoP = request.POST['algoritmoPrincipal']
        algoritmoC = request.POST['algoritmoComparar']
        dataset = request.POST['dataset']
        if algoritmoC == "Ninguno":
            context = {'algoritmoPrincipal': algoritmoP}
        else:
            context = {'algoritmoPrincipal': algoritmoP, 'algoritmoComparar': algoritmoC}
        return render(request, 'resultados.html', context)
    else:
        return render(request, 'resultados.html')

def neuronal(request):
    return render(request, 'neuronal.html')
