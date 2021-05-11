import pandas as pd
from modeling import modeling
from pre_procesado import preprocessed


def main():
    df_all = pd.read_csv('./datos.csv')

    # preprocesado
    df = preprocessed(df_all)

    # modelado
    modeling(df)


if __name__ == '__main__':
    main()
