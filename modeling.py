from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd


def decision_tree():
    """
    Entrena un árbol de decisión con x_train y y_train
    Devuelve un predicción de x_test
    """
    print("\nDECISION TREE\n")

    # Creamos el árbol
    return DecisionTreeClassifier()


def random_forest():
    """
    Entrena un random forest con x_train y y_train
    Devuelve un predicción de x_test
    """

    print("\nRANDOM FOREST\n")

    # Creamos el random_forest
    return RandomForestClassifier(n_estimators=100, random_state=1)


def svm():
    """
    Entrena un svm con x_train y y_train
    Devuelve un predicción de x_test
    """

    print("\nSVM\n")

    # Creamos el VSM
    return SVC(C=10.0)


def naive_bayes():
    """
    Entrena un Naive-Bayes con x_train y y_train
    Devuelve un predicción de x_test
    """

    print("\nNaive-Bayes\n")

    # Creamos el Naive-Bayes
    return GaussianNB()


class Model:
    decision_tree = decision_tree
    random_forest = random_forest
    svm = svm
    naive_bayes = naive_bayes


def compare_results(y_test, y_pred):
    """
    Compara los resultados clasificados por el modelo con los resultados reales
    """
    # Comparamos la predicción con los datos de testeo
    print(classification_report(y_test, y_pred))

    conf = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        columns=['Predicted relevant', 'Predicted no relevant'],
        index=['True relevant', 'True no relevant']
    )

    print(conf)


def modeling(df, model: Model):
    """
    Dividimos los datos en un 70% para entrenamiento y 30% para testeo,
    aparte separamos las variables de las etiquetas:
        - En X guardamos las variables
        - En Y las etiquetas
    """
    x_train, x_test, y_train, y_test = train_test_split(df.drop(['relevant'], axis=1), df['relevant'], test_size=0.3,
                                                        random_state=1)

    aux = model
    model = model()

    # Lo entremos
    cls = model.fit(x_train, y_train)

    if aux is Model.decision_tree:
        print("ENTRA")
        tree.plot_tree(cls)

    # Hacemos una predicción
    y_pred = model.predict(x_test)

    # comparamos resultados
    compare_results(y_test, y_pred)

