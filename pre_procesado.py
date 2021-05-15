import pandas as pd
from sklearn.impute import SimpleImputer


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

    # Sustitución de nulos por el más frecuente
    imputer_numericos = SimpleImputer(strategy='most_frequent')
    columnas = df.columns
    df = pd.DataFrame(imputer_numericos.fit_transform(df), columns=columnas)

    print(pd.isna(df).sum())

    return df
