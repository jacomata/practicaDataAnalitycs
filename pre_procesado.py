import pandas as pd


def preprocessed(df):
    print(df.info())  # Información del DataFrame

    print(df.describe())  # Descripción del DataGrame

    print(pd.isna(df).sum())  # Número de nulos de cada columna

    # Bucle para contar el número de ceros que tiene cada columna
    for column in df:
        contador = 0
        serie = df[column]
        for i in serie:
            if i == 0:
                contador += 1
        print(column, contador)

    return df