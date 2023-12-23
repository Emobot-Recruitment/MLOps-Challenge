# Challenge ML Ops - Recrutement Emobot

<br>

## Objectif du challenge

L’objectif de ce challenge est de prendre en main différents datasets couramment utilisés pour la détection de pose 2D et de proposer un **pipeline de traitement des données** permettant d’**entraîner un modèle simple** sur la fusion de ces datasets.

Une partie importante de votre travail consistera à traiter les données d'entraînement de chaque dataset de manière à les unifier selon un seul et même format afin de permettre l’entraînement d’un unique modèle sur tous les datasets à la fois.

Il vous faudra ainsi : 
- assurer la cohérence des données d’entrée (images)
- assurer la cohérence des labels (2D pose annotations)
- documenter votre démarche, choix et éventuels obstacles


*Dans ce travail, vous vous concentrerez uniquement sur les labels correspondants aux “2D pose annotations”.*

<br>

## Description des données

Votre travail portera sur les **4 datasets** suivants : 

- **MPII Human Pose**
	- Lien du dataset d’origine (~12Go)
http://human-pose.mpi-inf.mpg.de/
	- Lien du dataset pour le challenge (~100Mo)
https://drive.google.com/file/d/1vSJHetd8_GeLm7Ar6UKl9rlpOJ_l_eP6/view
- **Leeds Sports Pose (LSP)**
	- Doc : https://dbcollection.readthedocs.io/en/latest/datasets/leeds_sports_pose_extended.html 
	- Lien de téléchargement d’origine
https://datasets.activeloop.ai/docs/ml/datasets/lsp-dataset/ 

- **3D Poses in the Wild (3DPW)**
	- Lien du dataset d’origine (~5Go)
https://virtualhumans.mpi-inf.mpg.de/3DPW/ 
	- Lien du dataset pour le challenge (~65Mo)
https://drive.google.com/file/d/1WvHv3miZ2Y1bYB3eVMTw7oDuR8zubJFl/view


- **Multiperson Pose Test Set in 3D (MuPoTS-3D)**
	- Lien du dataset d’origine (~5Go)
http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/MultiPersonTestSet.zip 
	- Lien du dataset pour le challenge (~75Mo)
https://drive.google.com/file/d/1pwghLG6MY8OWkW2Qa6sjmls-Q209Kyvr/view

<br>

## Consignes détaillées

### Partie 1 : Prétraitement des Données pour l'Entraînement du Modèle

L'objectif principal de cette étape est de préparer les données issues de chaque dataset, en vue de leur utilisation pour l'entraînement du modèle fourni dans notre dépôt GitHub.

Vous devrez :

- **Téléchargement et Chargement des Données** : Accédez aux datasets mis à votre disposition et intégrez-les dans votre environnement de travail.
- **Analyse et Compréhension des Formats de Données** : Examinez la structure de chaque dataset. Votre capacité à comprendre et à interpréter le format des données sera essentielle.
- **Unification des Formats de Données** : Développez une stratégie pour harmoniser ces données dans un format unifié, adapté à l'entraînement du modèle.

Il est essentiel de justifier vos choix et d'expliciter votre compréhension des données (leur structure, leur influence sur le modèle, etc…). 

### Partie 2 : Model training

Un modèle jouet *SimpleBaseline*, vous est fourni dans le fichier `simple_baseline.py`. L’objectif est d’entraîner ce modèle sur le dataset fusionné que vous avez constitué dans la partie précédente. Un squelette de code d'entraînement vous est fourni dans le fichier `training.py`.

Vous êtes libre d’apporter des modifications au modèle ou bien de choisir une autre implémentation, du moment que ces choix sont motivés et explicités. Une attention particulière sera à nouveau apportée à la clarté de vos explications concernant les choix effectués et les éventuels problèmes rencontrés.


### Démarche attendue

Nous attendons de votre part une démarche **MLOps** : permettre à l’équipe d’ingénieurs IA de suivre l’entraînement des modèles, d’obtenir des métriques permettant son évaluation et de faciliter son utilisation future.

<br>

## Rendu

Vous devrez restituer votre travail sous la forme d’un **dossier .zip** contenant votre projet. Votre pipeline doit être restituée sous la forme d’un **Jupyter Notebook** nommé `main.ipynb`. Ce dernier devra être *abondamment commenté* afin de permettre à quiconque de prendre en main votre travail sans autre contexte. Il devra également pouvoir être lancé par une personne tierce et tourner sans accroc.
