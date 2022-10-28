# Introduction

Le projet de ce notebook a été réalisé dans le cadre de la formation d'[ingénieur machine learning proposé par Openclassrooms](https://openclassrooms.com/fr/paths/148-ingenieur-machine-learning).

Il portait sur la comparaison entres des modèles de computer vision entraînés initialement et l'utilisation de transfer learning.

La démarche a été réalisée de manière itérative par entraînements sucessifs de "nouveaux modèles":

* Modèle initial avec préprocessing seul
* Ajout d'optimisations : dropout, batchnormalization ...
* Implémentation de la data augmentation
* Transfer learning avec optimisation et data augmentation


L'ensemble des travaux ont été menés à l'aide du [dataset Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar). Ce dernier est constitué de 20 580 images de chiens triées en 120 classes relatives à leur race.

Les entrainements ont été réalisés sur GPU à l'aide de Google Colab.

# Contenu du repositiry:

Un notebook d'entraînement des modèles
Un script python de prédiction
Une présentation du projet

# Outils utilisés

* Jupyter Notebook/Google Colab Pro
* Python 3.8.5
* Numpy
* Matplotlib
* PIL
* OpenCV
* Tensorflow/Keras
