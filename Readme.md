
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

