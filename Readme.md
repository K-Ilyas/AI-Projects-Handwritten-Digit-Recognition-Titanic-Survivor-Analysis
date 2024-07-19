
# AI Projects: Handwritten Digit Recognition & Titanic Survivor Analysis

## Apprendre à reconnaître des nombres

Analyse du célèbre ensemble de données MNIST, qui comprend 70 000 images de chiffres manuscrits allant de 0 à 9. L’objectif principal est de développer des modèles de machine learning capables de classer ces chiffres avec précision.

### 1.1 Exploration des Données
L’analyse débute par une inspection du jeu de données, incluant la visualisation de plusieurs exemples de chiffres. Chaque chiffre est numérisé en une image de 28x28 pixels, où chaque pixel possède une valeur d’intensité variant de 0 à 255.

### 1.2 Prétraitement des Données
Nous divisons le jeu de données en ensembles d’entraînement et de test, et nous mélangeons les données d’entraînement pour assurer l'aléatoire. Par exemple, voici comment nous effectuons le mélange des données :

```python
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train.loc[shuffle_index], y_train.loc[shuffle_index]
```

De plus, nous créons des vecteurs cibles pour identifier le chiffre 5, ainsi que des vecteurs pour tous les chiffres (0 à 9).

### 1.3 Entraînement du Modèle
Nous entraînons un classificateur de descente de gradient stochastique (SGD) pour classer si un chiffre est un 5 ou non. Par exemple, voici comment nous entraînons ce classificateur :

```python
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train.values, y_train_5.values)
```

Le classificateur atteint un score de précision d’environ 96,4 % en utilisant une validation croisée. Nous évaluons également un classificateur "Jamais 5", qui prédit toujours que le chiffre n’est pas un 5, atteignant un score de précision d’environ 91 %.

### 1.4 Évaluation du Modèle
Nous évaluons le classificateur SGD entraîné en utilisant des métriques telles que la précision, le rappel et la matrice de confusion. Par exemple, voici comment nous obtenons la matrice de confusion :

```python
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
conf_matrix = confusion_matrix(y_train_5, y_train_pred)
print(conf_matrix)
```

Le score de précision est d’environ 84,9 % et le score de rappel est d’environ 77,5 %. La matrice de confusion fournit un aperçu de la performance du classificateur dans différentes classes.

### 1.5 Classification Multiclasse
Nous mettons en œuvre un classificateur multiclasse en utilisant le classificateur SGD, la stratégie "Un-contre-Un" (OvO) et le classificateur de forêt aléatoire. Par exemple, voici comment nous entraînons le classificateur OvO :

```python
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train.values, y_train)
```

Chaque classificateur est entraîné et évalué en utilisant une validation croisée. Le classificateur de forêt aléatoire atteint le meilleur score de précision d’environ 96,9 %.

### 1.6 Conclusion
Pour conclure, une série de modèles de machine learning ont été formés et évalués sur l’ensemble de données MNIST pour la tâche de classification des chiffres. Le modèle de forêt aléatoire s’est révélé être le plus performant. Des travaux futurs pourraient se concentrer sur l’optimisation et le réglage fin des modèles pour améliorer encore leur précision et efficacité.

<p align="center">
  <img height="400" width="600" src="https://github.com/user-attachments/assets/216980f9-2ab9-4cb7-a045-84fa54353a5b" alt>
  <br>
  <em>La solution du premier TP</em>
</p>


---

## Les survivants du Titanic

### 2.1 Importation et visualisation des données
Les données ont été importées à partir du fichier `train.csv` et les 10 premières lignes ont été affichées pour une première exploration.

```python
print(train.head(10))
```


### 2.2 Analyse des variables
#### 2.2.1 Variables à analyser
Les variables suivantes ont été analysées :

- Survived
- PClass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

#### 2.2.2 Résultats de l’analyse
1. Survived

- Signification : Indique si un passager a survécu (1) ou non (0) au naufrage du Titanic.
- Type : Variable binaire ou catégorielle.
- Valeurs manquantes : Aucune.


2. PClass

- Signification : Indique la classe du billet du passager (1ère, 2ème ou 3ème classe), un proxy pour la classe socio-économique.
- Type : Variable ordonnée ou catégorielle.
- Valeurs manquantes : Aucune.

3. Name

- Signification : Nom complet du passager, incluant le titre et le nom de famille.
- Type : Variable textuelle ou nominale.
- Valeurs manquantes : Aucune.
4. Sex

- Signification : Sexe du passager (male ou female).
- Type : Variable binaire ou catégorielle.
- Valeurs manquantes : Aucune.

5. Age

- Signification : Âge du passager en années.
- Type : Variable numérique ou continue.
- Valeurs manquantes : 177 valeurs manquantes (environ 20% des données).

6. SibSp

- Signification : Nombre de frères, sœurs ou conjoints à bord.
- Type : Variable numérique ou discrète.
- Valeurs manquantes : Aucune.
7. Parch

- Signification : Nombre de parents ou d’enfants à bord.
- Type : Variable numérique ou discrète.
- Valeurs manquantes : Aucune.
8. Ticket

- Signification : Numéro du billet du passager.
- Type : Variable textuelle ou nominale.
- Valeurs manquantes : Aucune.

9. Fare

- Signification : Prix du billet du passager en livres sterling.
- Type : Variable numérique ou continue.
- Valeurs manquantes : 1 valeur manquante (moins de 1% des données).
10. Cabin

- Signification : Numéro de cabine du passager.
- Type : Variable textuelle ou nominale.
- Valeurs manquantes : 687 valeurs manquantes (environ 77% des données).
11. Embarked

- Signification : Port d’embarquement du passager (C = Cherbourg, Q = Queenstown, S = Southampton).
- Type : Variable catégorielle ou nominale.
- Valeurs manquantes : 2 valeurs manquantes (moins de 0.22% des données).
### 2.3 Modélisation
#### 2.3.1 Préparation des données
Les variables explicatives utilisées pour le premier modèle sont 'Fare', 'SibSp' et 'Parch'.

```python
target = train[['Fare', 'SibSp', 'Parch']]
```
#### 2.3.2 Note: interprétation concernant les variables du modèle LogisticRegression
Les coefficients de la régression logistique permettent d’interpréter l’importance des variables explicatives dans le modèle. Plus un coefficient est grand en valeur absolue, plus la variable correspondante est importante dans le modèle. Les variables explicatives les plus importantes dans le modèle sont les variables Pclass 1 et Pclass 3, qui sont les variables indicatrices de la variable Pclass. Ces variables sont les plus importantes car elles ont les coefficients les plus grands en valeur absolue. Elles sont donc les variables les plus discriminantes pour prédire la survie des passagers du Titanic. Les variables SibSp, Parch et Fare sont moins importantes, car elles ont des coefficients plus petits en valeur absolue. Elles sont donc moins discriminantes pour prédire la survie des passagers du Titanic.

#### 2.3.3 Stratégie de validation
Une validation croisée à 5 plis a été utilisée pour évaluer la performance des modèles.

##### 2.3.4 Modèle initial
Un modèle de régression logistique a été utilisé comme premier modèle. Le score de validation croisée obtenu est de 0.676.

<p align="center">
  <img height="100" width="400" src="https://github.com/user-attachments/assets/cb54fbfd-9163-445c-bc50-a12fad8ab328" alt>
  <br>
  <em>La solution du deuxième TP</em>
</p>

#### 2.3.5 Amélioration du modèle
1. Ajout de la variable 'Pclass' :
En ajoutant la variable 'Pclass' au modèle, le score de validation croisée a augmenté à 0.681. Les coefficients de la régression logistique ont montré que les variables indicatrices de 'Pclass' sont importantes pour prédire la survie.

2. Ajout des variables 'Sex' et 'Age' :
En ajoutant les variables 'Sex' et 'Age' au modèle, le score de validation croisée a augmenté à 0.820. Les coefficients ont montré que 'Sex' est une variable très importante dans la prédiction de la survie.

3. Ajout de la variable 'is child' :
En ajoutant la variable 'is child' (indicatrice si le passager est un enfant) au modèle, le score de validation croisée est resté à 0.820.

#### 2.3.6 Modèles alternatifs
- Random Forest :
Un classifieur Random Forest a été utilisé pour construire un modèle alternatif. Le score obtenu était supérieur à celui de la régression logistique.

- Gradient Boosting Classifier :
Un Gradient Boosting Classifier a également été utilisé pour construire un modèle alternatif. Le score obtenu était de 0.837.

- Support Vector Machine (SVM) :
Les SVM sont des modèles d’apprentissage supervisé utilisés pour la classification et la régression. Ils cherchent à trouver un hyperplan optimal qui sépare les données en classes. Les SVM sont efficaces pour les problèmes de classification binaire et peuvent également être étendus à la classification multiclasse. Ils sont sensibles au choix du noyau (linéaire, polynomial, RBF, etc.). Un SVM avec un noyau linéaire a été utilisé pour construire un autre modèle alternatif. Le score obtenu était de 0.803.

## 2.4 Nouvelles variables
Boosting (Renforcement) : Le boosting est une technique d’apprentissage automatique qui combine plusieurs modèles faibles (par exemple, des arbres de décision peu profonds) pour créer un modèle plus puissant. Il fonctionne en ajustant itérativement les poids des observations mal classées, ce qui permet d’améliorer la précision du modèle global. Les algorithmes de boosting populaires incluent AdaBoost, Gradient Boosting et XGBoost. Des nouvelles variables 'Title' (titre dans le nom) et 'Cabin' ont été créées à partir des données existantes. En les ajoutant au modèle, le score de validation croisée a augmenté à 0.820.

## 2.5 Conclusion

À partir de l’analyse des variables et de la modélisation des données du Titanic, plusieurs conclusions peuvent être tirées :

Les variables 'Sex' et 'Pclass' sont très importantes pour prédire la survie des passagers.
<br>
L’ajout de nouvelles variables telles que 'Title' et 'Cabin' peut améliorer la performance du modèle.
<br>
Les modèles de type Gradient Boosting Classifier ont donné les meilleurs scores de prédiction.
<br>
Des efforts supplémentaires pourraient être faits pour explorer d’autres variables potentiellement importantes et pour affiner les modèles existants afin d’obtenir de meilleures performances de prédiction.
### Plots :
---

<p align="center">
  <img height="400" width="800" src="https://github.com/user-attachments/assets/d3194ea0-de4d-4659-85dd-5a9bb23ed1f2" alt>
  <br>
  <em>Plot 1</em>
</p>



<p align="center">
  <img height="400" width="800" src="https://github.com/user-attachments/assets/2aae11de-f738-420f-8de8-e5c75073fa8c" alt>
  <br>
  <em>Plot 2</em>
</p>


<p align="center">
  <img height="400" width="800" src="https://github.com/user-attachments/assets/70ae4e9d-ae17-419c-a54d-3ada893aeaf6" alt>
  <br>
  <em>Plot 3</em>
</p>


<p align="center">
  <img height="400" width="800" src="https://github.com/user-attachments/assets/24141ce5-e93a-49b6-9092-9850d33db0e0" alt>
  <br>
  <em>Plot 4</em>
</p>



<p align="center">
  <img height="400" width="800" src="https://github.com/user-attachments/assets/e7e61a61-a019-4246-8ee8-b22b78fc3f24" alt>
  <br>
  <em>Plot 5</em>
</p>


<p align="center">
  <img height="400" width="800" src="https://github.com/user-attachments/assets/9f745bf4-0f6c-4024-b20d-aeaf8661266b" alt>
  <br>
  <em>Plot 6</em>
</p>

### Demo 

(https://replit.com/@K-Ilyas/AI-Projects-Handwritten-Digit-Recognition-Titanic-Survivor-A)[Demo]





