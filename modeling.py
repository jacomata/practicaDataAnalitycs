from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


def modeling(df):
    """
    Dividimos los datos en un 70% para entrenamiento y 30% para testeo,
    aparte separamos las variables de las etiquetas:
        - En X guardamos las variables
        - En Y las etiquetas
    """
    x_train, x_test, y_train, y_test = train_test_split(df.drop(['relevant'], axis=1), df['relevant'], test_size=0.3)

    # Creamos el árbol
    tree = DecisionTreeClassifier()

    # Lo entremos
    tree.fit(x_train, y_train)

    # Hacemos una predicción
    y_pred = tree.predict(x_test)

    # Comparamos la predicción con los datos de testeo
    print(classification_report(y_test, y_pred))

    # Comparamos los reales con los predichos

    conf = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        columns=['Predicted relevant', 'Predicted X'],
        index=['True relevant', 'True X']
    )

    print(conf)
