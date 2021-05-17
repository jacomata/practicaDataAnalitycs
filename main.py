import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from modeling import decision_tree, compare_results, Model, modeling
from pre_procesado import preprocessed


def main():
    """
    Función principal
    """
    df_all = pd.read_csv('./datos.csv')

    # print(df_all.info())
    df_all.drop(['inter_parts'], inplace=True, axis=1)
    # print(df_all.info())

    # preprocesado
    df = preprocessed(df_all)

    modeling(df, Model.decision_tree)
    modeling(df, Model.random_forest)
    modeling(df, Model.svm)
    modeling(df, Model.naive_bayes)


def export_model():
    df = pd.read_csv('./datos.csv')
    model = DecisionTreeClassifier(min_samples_split=4, min_samples_leaf=4, max_depth=4)
    x_train, x_test, y_train, y_test = train_test_split(df.drop(['relevant'], axis=1), df['relevant'], test_size=0.0001,
                                                        random_state=1)

    model.fit(x_train, y_train)

    print(y_test)
    """
    with open('model.pkl','wb') as model_file:
    pickle.dump()
    """


if __name__ == '__main__':
    export_model()
