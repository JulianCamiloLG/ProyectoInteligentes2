# Generated by Django 2.2.2 on 2019-08-13 03:28

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DataSet',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('dataset', models.FileField(blank=True, upload_to='datasets/')),
            ],
        ),
        migrations.CreateModel(
            name='Index',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('algoritmoPrincipal', models.CharField(choices=[('id3', 'ID3'), ('apriori', 'Apriori'), ('kmeans', 'K-Means'), ('knn', 'KNN'), ('fpgrouth', 'FP-Grouth'), ('regresion', 'Regresión Lineal')], default='', max_length=15)),
                ('compararCon', models.CharField(blank=True, choices=[('id3', 'ID3'), ('apriori', 'Apriori'), ('kmeans', 'K-Means'), ('knn', 'KNN'), ('fpgrouth', 'FP-Grouth'), ('regresion', 'Regresión Lineal')], max_length=15)),
                ('selectDataSet', models.ForeignKey(blank=True, on_delete=django.db.models.deletion.DO_NOTHING, to='algoritmos.DataSet')),
            ],
        ),
    ]
