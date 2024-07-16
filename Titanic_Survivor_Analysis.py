from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt


pd.options.mode.chained_assignment = None


# 1.1 Importation des données :

train = pd.read_csv('./train.csv', sep=',')

#  1.2. Visualisation :
# Question 1.2 : Visualiser les 10 premières lignes du fichier (à analyser )

print(train.head(10))

#  1.3. Indexation des données :

train.set_index('PassengerId', inplace=True, drop=True)

#  1.4. Analyse des variables

# Question 1.4.1 : Ecrire la commande qui permet de visualiser les noms.

print(train.columns)

#  Question 1.4.2 :  Faites une analyse des variables suivantes : Survived, PClass, Name, Sex, Age, SibSp,Parch, Ticket, Fare, Cabin, Embarked.
'''
Survived: Cette variable indique si un passager a survécu ou non au naufrage du Titanic. Elle prend la valeur 1 pour les survivants et 0 pour les victimes. C’est une variable binaire ou catégorielle. Il n’y a pas de valeurs manquantes pour cette variable.

train[train["Survived"].isnull()].shape[0] = 0

========================================================================================================================================================

Pclass: Cette variable indique la classe du billet du passager (1ère, 2ème ou 3ème classe). Elle est un proxy pour la classe socio-économique du passager. C’est une variable ordonnée ou catégorielle. Il n’y a pas de valeurs manquantes pour cette variable.

train[train["PClass"].isnull()].shape[0] = 0

========================================================================================================================================================

Name: Cette variable indique le nom complet du passager, y compris le titre et le nom de famille. C’est une variable textuelle ou nominale. Il n’y a pas de valeurs manquantes pour cette variable.

train[train["Name"].isnull()].shape[0] = 0

========================================================================================================================================================

Sex: Cette variable indique le sexe du passager (male ou female). C’est une variable binaire ou catégorielle. Il n’y a pas de valeurs manquantes pour cette variable.

train[train["Sex"].isnull()].shape[0] = 0

========================================================================================================================================================

Age: Cette variable indique l’âge du passager en années, avec certaines valeurs fractionnaires pour les nourrissons. C’est une variable numérique ou continue. Il y a 177 valeurs manquantes pour cette variable, soit environ 20% des données.

train[train["Age"].isnull()].shape[0] = 177

========================================================================================================================================================

SibSp: Cette variable indique le nombre de frères, sœurs ou conjoints du passager à bord du Titanic. C’est une variable numérique ou discrète. Il n’y a pas de valeurs manquantes pour cette variable.

train[train["SibSp"].isnull()].shape[0] = 0

========================================================================================================================================================

Parch: Cette variable indique le nombre de parents ou d’enfants du passager à bord du Titanic. C’est une variable numérique ou discrète. Il n’y a pas de valeurs manquantes pour cette variable.

train[train["Parch"].isnull()].shape[0] = 0

========================================================================================================================================================

Ticket: Cette variable indique le numéro du billet du passager. C’est une variable textuelle ou nominale. Il n’y a pas de valeurs manquantes pour cette variable.

train[train["Ticket"].isnull()].shape[0] = 0

========================================================================================================================================================

Fare: Cette variable indique le prix du billet du passager en livres sterling. C’est une variable numérique ou continue. Il y a 1 valeur manquante pour cette variable, soit moins de 1% des données.

train[train["Fare"].isnull()].shape[0] = 0

========================================================================================================================================================

Cabin: Cette variable indique le numéro de la cabine du passager. C’est une variable textuelle ou nominale. Il y a 687 valeurs manquantes pour cette variable, soit environ 77% des données.

train[train["Cabin"].isnull()].shape[0] = 687

========================================================================================================================================================

Embarked: Cette variable indique le port d’embarquement du passager (C = Cherbourg, Q = Queenstown, S = Southampton). C’est une variable catégorielle ou nominale. Il y a 2 valeurs manquantes pour cette variable, soit moins de 0.22% des données.

train[train["Embarked"].isnull()].shape[0] = 0

'''

#  Question 1.4.3  : Utiliser respectivement les commandes dtype et count qui vont vous donner  respectivement le type des données et le nombre de valeurs non nulles par variables.

print(train.dtypes)

print(train.count())

# 2. Premier Modèle :

#  2.1. Préparation des données

target = train[['Fare', 'SibSp', 'Parch']]


print(target)


# Question 2.2.1 : Ecrire une fonction  compute_score qui prend comme paramètre un classifieur et un jeu de données et qui renvoie le score de validation croisée du classifieur sur le jeu de données.


def parse_model_1(file):

    train = pd.read_csv(file, sep=',')

    train.set_index('PassengerId', inplace=True, drop=True)

    y = train['Survived']

    X = train[['Fare', 'SibSp', 'Parch']]

    print(X)
    return X, y


#  2.2. Stratégie de Validation

'''

Une validation croisée est une méthode qui permet d’évaluer la performance et la robustesse d’un modèle d’apprentissage sur des données inédites.
Elle consiste à diviser le jeu de données en plusieurs sous-ensembles, et à utiliser successivement l’un d’entre eux comme données de test, et les autres comme données d’entraînement. 
Ainsi, on obtient plusieurs estimations de l’erreur de généralisation du modèle, que l’on peut moyenne pour avoir une mesure plus fiable.

'''

#  Question 2.2.2 : Ecrire une fonction  compute_score qui prend comme paramètre un classifieur et un jeu de données et qui renvoie le score de validation croisée du classifieur sur le jeu de données.


def compute_score(classifieur, X, y):

    return cross_val_score(classifieur, X, y, cv=5).mean()


#  Question 2.2.3 : Calculer le modèle de regression logistique (qui constituera votre 1er modèle de prédiction) et utiliser la fonction compute_score de la question 2.2.2 pour obtenir le score qui indique la qualité de la prédiction. Ce score sera le score à battre avec les autres modèles que vous allez écrire.

model = LogisticRegression()

X, y = parse_model_1('./train.csv')


X = X[:int(X.shape[0] * 0.9)]
y = y[:int(y.shape[0] * 0.9)]

model.fit(X, y)


print((model.predict(X[int(X.shape[0] * 0.9):])
      == y[int(y.shape[0] * 0.9):]).value_counts())

print(compute_score(model, X, y))


# Question 3.1.1 : Vous allez tout d’abord séparer les données en deux populations, les survivants et les victimes. avec les instructions suivantes :

survived = train[train.Survived == 1]
dead = train[train.Survived == 0]


#  Question 3.1.2 : Ecrire la fonction plot_hist qui prend en arguments le nom d’une des variables des données (Pclass ou autres) et qui trace l’histogramme de la distribution des survivants et des victimes pour la variable passée en paramètre..

def plot_hist(variable):
    plt.hist([dead[variable].dropna(), survived[variable].dropna()],
             stacked=False, color=['b', 'purple'], bins=30, label=['Victimes', 'Survivant'])
    plt.xlabel(variable)
    plt.ylabel('Distribution relative de '.format(variable))
    plt.legend()
    plt.show()


#  Question 3.1.3 : Utiliser la fonction plot_hist pour tracer l’histogramme de la distribution des survivants et des victimes pour la variable Pclass.

plot_hist('Pclass')


# Question 3.2.2 : La variable Pclass est une variable importante. Pourquoi ?

'''

La variable Pclass est une variable importante car elle reflète le statut socio-économique des passagers du Titanic.
Elle est un bon indicateur du sort des passagers, car les passagers de 1ère classe ont eu un taux de survie plus élevé que les passagers de 2ème classe, 
qui ont eux-mêmes eu un taux de survie plus élevé que les passagers de 3ème classe.
On peut donc discriminer les survivants et les victimes selon la classe de voyage,
car il existe une corrélation entre la classe et le taux de survie. 
La variable Pclass est donc un bon indicateur du sort des passagers du Titanic.

'''


#  3.3 Traitement de la variable Pclass :

#  Question 3.3.1 : Ecrire une fonction parse_model_2 qui prend en argument le nom du fichier et qui renvoie les variables explicatives X et la variable cible y.

def parse_model_2(file):

    train = pd.read_csv(file, sep=',')

    train.set_index('PassengerId', inplace=True, drop=True)

    y = train['Survived']

    X = train[['SibSp', 'Parch', 'Fare', 'Pclass']]

    X = pd.concat(
        [X, pd.get_dummies(X['Pclass'], prefix='Pclass', dtype=int)], axis=1)
    X.drop('Pclass', axis=1, inplace=True)

    return X, y


# Question 3.4.1 : Calculer le modèle de régression logistique avec ce deuxième modèle et calculer le score de validation croisée avec ce nouveau modèle :


model = LogisticRegression(max_iter=10000)

X, y = parse_model_2('./train.csv')


X = X[:int(X.shape[0] * 0.9)]
y = y[:int(y.shape[0] * 0.9)]

model.fit(X, y)


print((model.predict(X[int(X.shape[0] * 0.9):])
      == y[int(y.shape[0] * 0.9):]).value_counts())

print(compute_score(model, X, y))


#  Question 3.4.2 : Que remarquez vous? (indication : a-t-on améliorer le 1er modèle?)

'''
    Le score de validation croisée du modèle de régression logistique avec la variable Pclass est de 0.681.
    Ce score est superier à celui du premier modèle, qui était de 0.676.

    On a donc amélioré le premier modèle en ajoutant la variable Pclass.'''

# Question 3.4.3 : Tracer les coefficients de la régression logistique en écrivant une fonction qui affiche les coefficients : plot_lr_coefs(X, lr) où X est la matrice des données d’entrainement et lr un classifieur de régression logistique. vous devez obtenir un graphique de ce type.


def plot_lr_coefs(X, lr):
    plt.figure(figsize=(10, 5))
    plt.bar(X.columns, lr.coef_[0])
    plt.xticks(rotation=90)
    plt.show()


plot_lr_coefs(X, model)

# Question 3.4.4 : Donner une interprétation concernant les variables du modèle 2 à partir du graphique de la question 3.4.3

'''
    Les coefficients de la régression logistique permettent d’interpréter l’importance des variables explicatives dans le modèle.
    Plus un coefficient est grand en valeur absolue, plus la variable correspondante est importante dans le modèle.
    Les variables explicatives les plus importantes dans le modèle sont les variables Pclass_1 et Pclass_3, qui sont les variables indicatrices de la variable Pclass.  
    Ces variables sont les plus importantes car elles ont les coefficients les plus grands en valeur absolue.
    Elles sont donc les variables les plus discriminantes pour prédire la survie des passagers du Titanic.
    Les variables SibSp, Parch et Fare sont moins importantes, car elles ont des coefficients plus petits en valeur absolue.
    Elles sont donc moins discriminantes pour prédire la survie des passagers du Titanic.
    
    '''

#  4.1. Création d’un nouveau jeu de données avec les variable Sex et Age :
#  Question 4.1 : Ecrire la fonction parse_model_3 en injectant les instructions permettant s’insérer les variables Age et Sex  dans la fonction parse_model_2 de la question 3.3.1


def parse_model_3(file):

    train = pd.read_csv(file, sep=',')

    train.set_index('PassengerId', inplace=True, drop=True)

    y = train['Survived']

    X = train[['SibSp', 'Parch', 'Fare', 'Pclass', 'Age', 'Sex']]

    X['Age'] = X['Age'].fillna(X['Age'].median())

    X = pd.concat(
        [X, pd.get_dummies(X['Pclass'], prefix='Pclass', dtype=int)], axis=1)
    X.drop('Pclass', axis=1, inplace=True)

    X = pd.concat(
        [X, pd.get_dummies(X['Sex'], prefix='Sex', dtype=int)], axis=1)
    X.drop('Sex', axis=1, inplace=True)

    return X, y


# Question 4.2.1 : Utiliser la régression logistique comme précédemment pour calculer le nouveau modèle et calculer le nouveau score.

model = LogisticRegression(max_iter=10000)

X, y = parse_model_3('./train.csv')


X = X[:int(X.shape[0] * 0.9)]
y = y[:int(y.shape[0] * 0.9)]

model.fit(X, y)


print(compute_score(model, X, y))


#  Question 4.2.2 : Afficher le graphique permettant d’analyser le poids des variables du modèle 2.
plot_lr_coefs(X, model)

#  Question 4.2.3 : quelles conclusions faites vous pour ce qui concerne l’amélioration du modèle (est ce due à la variable  Age ou aux variable de type Sex?

''' L’amélioration du modèle est principalement due à l’ajout des variables de type Sex, qui ont un poids important dans la régression logistique. '''


#  Question 4.3.1 : Tracer les histogrammes des survivants et des victimes pour la variable Age.

plot_hist('Age')


# Question 4.3.3 : Ajouter cette instruction pour obtenir un nouveau modèle permettant de prendre en compte la variable is_child.


# parse model 4

def parse_model_4(file):

    train = pd.read_csv(file, sep=',')

    train.set_index('PassengerId', inplace=True, drop=True)

    y = train['Survived']

    X = train[['SibSp', 'Parch', 'Fare', 'Pclass', 'Age', 'Sex']]

    X['Age'] = X['Age'].fillna(X['Age'].median())

    X['is_child'] = X['Age'] < 8

    X = pd.concat(
        [X, pd.get_dummies(X['Pclass'], prefix='Pclass', dtype=int)], axis=1)
    X.drop('Pclass', axis=1, inplace=True)

    X = pd.concat(
        [X, pd.get_dummies(X['Sex'], prefix='Sex', dtype=int)], axis=1)
    X.drop('Sex', axis=1, inplace=True)

    return X, y


#  Question 4.3.5 :  Vérifier que le score obtenu correspondant  à une amélioration est du à l’introduction de la variable is_child (pour cela utiliser la fonction plot_lr_coefs(X, lr) de la question 3.4.3).

model = LogisticRegression(max_iter=10000)

X, y = parse_model_4('./train.csv')


X = X[:int(X.shape[0] * 0.9)]
y = y[:int(y.shape[0] * 0.9)]

model.fit(X, y)


print(compute_score(model, X, y))


plot_lr_coefs(X, model)


# Question 5.1.1 : Appliquer le classifieur RandomForest de scikit learn sur le modèle sans la variable child et calculer le nouveau score obtenu.


model = RandomForestClassifier()

X, y = parse_model_3('./train.csv')


X = X[:int(X.shape[0] * 0.9)]
y = y[:int(y.shape[0] * 0.9)]

model.fit(X, y)

print(compute_score(model, X, y))

#  Question 5.1.2 : Que remarquez vous?

''' le score obtenu avec le classifieur Random Forest est supérieur à celui obtenu avec la régression logistique. 
Cela peut s’expliquer par le fait que le modèle non linéaire est capable de capturer des interactions entre les variables, ou de modéliser des effets non monotones de certaines variables sur la survie.'''


def classifier_importance(X, clf):
    import numpy as np
    import pylab as pl

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    pl.title("Feature importances")

    for tree in clf.estimators_:
        pl.plot(range(X.shape[1]), tree.feature_importances_[indices], 'r')

    pl.plot(range(X.shape[1]), importances[indices], 'b')
    pl.show()

    for f in range(X.shape[1]):
        print("%d. feature: %s (%f)" %
              (f + 1, X.columns[indices[f]], importances[indices[f]]))


# Question 5.2.1 : Utiliser la fonction classifier_importance(X, clf) après avoir créer un classifieur de type Random Forest puis l’entrainer.
model = RandomForestClassifier(n_estimators=10)

X, y = parse_model_4('./train.csv')

X = X[:int(X.shape[0] * 0.9)]
y = y[:int(y.shape[0] * 0.9)]

model.fit(X, y)

classifier_importance(X, model)

#  Question 5.3.1 : Créer une nouvelle variable Title avec l’instruction suivante :


#  Question 5.3.2 : Créer une nouvelle variable cabin avec l’instruction suivante :


def parse_model_5(file):

    train = pd.read_csv(file, sep=',')

    train.set_index('PassengerId', inplace=True, drop=True)

    y = train['Survived']

    X = train[['SibSp', 'Parch', 'Fare', 'Pclass', 'Age', 'Sex']]

    X['Age'] = X['Age'].fillna(X['Age'].median())

    X['is_child'] = X['Age'] < 8

    X['title'] = train['Name'].map(lambda x: x.split(',')[1].split('.')[0])
    X['cabin'] = train['Cabin'].map(lambda x: x[0] if not pd.isnull(x) else -1)

    X = pd.concat(
        [X, pd.get_dummies(X['Pclass'], prefix='Pclass', dtype=int)], axis=1)
    X.drop('Pclass', axis=1, inplace=True)

    X = pd.concat(
        [X, pd.get_dummies(X['title'], prefix='title', dtype=int)], axis=1)
    X.drop('title', axis=1, inplace=True)

    X = pd.concat(
        [X, pd.get_dummies(X['cabin'], prefix='cabin', dtype=int)], axis=1)
    X.drop('cabin', axis=1, inplace=True)

    X = pd.concat(
        [X, pd.get_dummies(X['Sex'], prefix='Sex', dtype=int)], axis=1)
    X.drop('Sex', axis=1, inplace=True)

    return X, y

# Question 5.3.4 : Calculer le modèle de régression logistique correspond et calcul le score obtenu


model = LogisticRegression(max_iter=10000)

X, y = parse_model_5("./train.csv")
X = X[:int(X.shape[0] * 0.9)]
y = y[:int(y.shape[0] * 0.9)]

model.fit(X, y)


print(compute_score(model, X, y))

# Question 5.3.5 : Que remarquez vous?

'''
 Le score de validation croisée du modèle de régression logistique avec la variable Title et cabin est de 0.82.
 Ce score est superier à celui du premier modèle, qui était de 0.80.
 On a donc amélioré le premier modèle en ajoutant la variable title et cabin.
'''

#  6.1 Améliorer le modèle de régression logistique

#  Question 6.1.1  : Vous pouvez aussi étudier les autres variables :  surname, embarked, ou ticket éventuellement.


def parse_model_6(file):

    train = pd.read_csv(file, sep=',')

    train.set_index('PassengerId', inplace=True, drop=True)

    y = train['Survived']

    X = train[['SibSp', 'Parch', 'Fare', 'Pclass',
               'Age', 'Sex', 'Embarked', 'Ticket']]

    X['Age'] = X['Age'].fillna(X['Age'].median())

    X['Embarked'] = X['Embarked'].fillna(X['Age'].median())

    X['is_child'] = X['Age'] < 8

    X['title'] = train['Name'].map(lambda x: x.split(',')[1].split('.')[0])
    X['cabin'] = train['Cabin'].map(lambda x: x[0] if not pd.isnull(x) else -1)

    X = pd.concat(
        [X, pd.get_dummies(X['Pclass'], prefix='Pclass', dtype=int)], axis=1)
    X.drop('Pclass', axis=1, inplace=True)

    X = pd.concat(
        [X, pd.get_dummies(X['Ticket'], prefix='Ticket', dtype=int)], axis=1)
    X.drop('Ticket', axis=1, inplace=True)
    X = pd.concat(
        [X, pd.get_dummies(X['Embarked'], prefix='Embarked', dtype=int)], axis=1)
    X.drop('Embarked', axis=1, inplace=True)

    X = pd.concat(
        [X, pd.get_dummies(X['title'], prefix='title', dtype=int)], axis=1)
    X.drop('title', axis=1, inplace=True)

    X = pd.concat(
        [X, pd.get_dummies(X['cabin'], prefix='cabin', dtype=int)], axis=1)
    X.drop('cabin', axis=1, inplace=True)

    X = pd.concat(
        [X, pd.get_dummies(X['Sex'], prefix='Sex', dtype=int)], axis=1)
    X.drop('Sex', axis=1, inplace=True)

    return X, y


X, y = parse_model_6("./train.csv")


model = LogisticRegression(max_iter=10000)

X, y = parse_model_5("./train.csv")
X = X[:int(X.shape[0] * 0.9)]
y = y[:int(y.shape[0] * 0.9)]

model.fit(X, y)

print(compute_score(model, X, y))


# 6.2 Nouveaux modèles

'''
Boosting (Renforcement) : Le boosting est une technique d’apprentissage automatique qui combine plusieurs modèles faibles (par exemple, des arbres de décision peu profonds) pour créer un modèle plus puissant.
Il fonctionne en ajustant itérativement les poids des observations mal classées, ce qui permet d’améliorer la précision du modèle global. 
Les algorithmes de boosting populaires incluent AdaBoost, Gradient Boosting et XGBoost.
'''


X, y = parse_model_5("./train.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

gb_classifier = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

gb_classifier.fit(X_train, y_train)

y_pred = gb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("score : ", accuracy)


'''
Support Vector Machines (SVM) : Les SVM sont des modèles d’apprentissage supervisé utilisés pour la classification et la régression. 
Ils cherchent à trouver un hyperplan optimal qui sépare les données en classes. 
Les SVM sont efficaces pour les problèmes de classification binaire et peuvent également être étendus à la classification multiclasse.
Ils sont sensibles au choix du noyau (linéaire, polynomial, RBF, etc.).
'''


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("score : ", accuracy)
