
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
  <img height="200" width="600" src="https://github.com/user-attachments/assets/0a06ee6f-0ea8-49af-82fd-d46d61394f57" alt>
  <br>
  <em>La solution du deuxième TP</em>
</p>

