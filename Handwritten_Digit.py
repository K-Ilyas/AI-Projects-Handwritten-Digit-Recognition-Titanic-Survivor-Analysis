
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


mnist = fetch_openml('mnist_784')


# Question 1 : représente simplement l’intensité d’un pixel  :

X, y = mnist["data"], mnist["target"]

# print(X)
# print(y.shape)
# y = y.astype(np.int8)


some_digit = X.loc[25997]
some_digit_value = y.loc[25997]
print("data : ", some_digit, "target :",
      some_digit_value)  # le nombre obtenu est 5
some_digit_image = some_digit.values.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")
plt.show()


# Question 2 Quelques chiffres de la base de données MNIST :

X_train, X_test, y_train, y_test = X.loc[:
                                         60000], X.loc[60000:], y.loc[:60000], y.loc[60000:]

# cast to int

y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)


# Question : Effectuer des permutations avec le code suivant :

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train.loc[shuffle_index], y_train.loc[shuffle_index]

# Question 3 : Entrainement d’un classificateur
# On commence par créer 2 vecteurs cibles pour cette tache d’identification:

# vrai pour les 5, faux pour le reste.

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Creation d’un classifier SGD et entrainement :

sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train.values, y_train_5.values)


# prediction

print(sgd_clf.predict([some_digit.values]))


# Question 4  : Mesures de performance


# result :  [0.96465 0.96635 0.96135]
print(cross_val_score(sgd_clf, X_train.values,
      y_train_5, cv=3, scoring="accuracy"))


# Question 4.2: Donner taux d’exactitude de ce modèle ?

never_5_clf = Never5Classifier()

print(cross_val_score(never_5_clf, X_train.values, y_train_5, cv=3,
      scoring="accuracy"))  # [0.9075  0.9097  0.91175]


########################### Question 5 : Matrice de Confusion ####################################


# confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

print(confusion_matrix(y_train_5, y_train_pred))


# Question 51. :  Utiliser ScikitLearn pour calculer les métrique précision et rappel.

print(precision_score(y_train_5, y_train_pred))

print(recall_score(y_train_5, y_train_pred))


# Question 6.1 Implémentation d’un classifier multiclasses :

sgd_clf.fit(X_train.values, y_train)
print(sgd_clf.predict([some_digit.values]))

# Question 6.2 Classifieur multiClasses OvO

ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train.values, y_train)
print(ovo_clf.predict([some_digit.values]))
print(len(ovo_clf.estimators_))

# Question 6.3  Classifieur basé Foret Aléatoire.
# Question 6.3.1. Entrainement d’un RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train.values, y_train)
print(forest_clf.predict([some_digit.values]))

# Question 6.3.2. Obtention des probabilités attribuées aux différentes classes.

print(forest_clf.predict_proba([some_digit.values]))


# Question 6.4. Evaluation des classifiers

print(cross_val_score(sgd_clf, X_train.values, y_train, cv=3, scoring="accuracy"))
