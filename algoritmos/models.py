from django.db import models


# Create your models here.
class DataSet(models.Model):
    dataset = models.FileField(upload_to='datasets/', blank=True)

    def __str__(self):
        return self.dataset


class Index(models.Model):
    algoritmos = (('id3', 'ID3'), ('apriori', 'Apriori'), ('kmeans', 'K-Means'), ('knn', 'KNN'),
                  ('fpgrouth', 'FP-Grouth'), ('regresion', 'Regresión Lineal'))
    algoritmoPrincipal = models.CharField(max_length=15, choices=algoritmos, default="")
    selectDataSet = models.ForeignKey(DataSet, blank=True, on_delete=models.DO_NOTHING)
    compararCon = models.CharField(max_length=15, choices=algoritmos, blank=True)

    def __str__(self):
        return self.algoritmoPrincipal

#class RedNeuronal(models.Model):
#s    categorias =