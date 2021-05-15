import pandas as pd
from sklearn.model_selection import train_test_split

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


if __name__ == '__main__':
    main()
