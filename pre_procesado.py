import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def preprocessed(df):
    # print(df.info())  # Información del DataFrame
    # print(df.describe())  # Descripción del DataGrame
    # print(pd.isna(df).sum())  # Número de nulos de cada columna

    # Bucle para contar el número de ceros que tiene cada columna
    for column in df:
        contador = 0
        serie = df[column]
        for i in serie:
            if i == 0:
                contador += 1
        # print(column, contador)

    # print(pd.isna(df).sum())

    # Añadimos el DataFrame a un archivo xlsx para poder visualizarlos mejor
    df.to_excel('./datos_en_excel.xlsx')

    # Tratamiento de outliers
    for col_name in df.select_dtypes(include=np.number).columns[:-1]:
        # print(col_name)
        q1 = df[col_name].quantile(0.25)
        q3 = df[col_name].quantile(0.75)
        iqr = q3 - q1

        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        # print("Outliers = ", df.loc[(df[col_name] < low) | (df[col_name] > high), col_name])

    # Script to exclude the outliers
    for col_name in df.select_dtypes(include=np.number).columns[:-1]:
        # print(col_name)
        q1 = df[col_name].quantile(0.25)
        q3 = df[col_name].quantile(0.75)
        iqr = q3 - q1

        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        # print("Exclude the Outliers = ", df.loc[~((df[col_name] < low) | (df[col_name] > high)), col_name])
        df[col_name] = df.loc[~((df[col_name] < low) | (df[col_name] > high)), col_name]

    for col_name in df.select_dtypes(include=np.number).columns[:-1]:
        # print(col_name)
        q1 = df[col_name].quantile(0.25)
        q3 = df[col_name].quantile(0.75)
        iqr = q3 - q1

        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        # print("Change the outliers with median ", df[col_name].median())
        df.loc[(df[col_name] < low) | (df[col_name] > high), col_name] = df[col_name].median()

    # Sustitución de nulos por el más frecuente
    imputer_numericos = SimpleImputer(strategy='most_frequent')
    columnas = df.columns
    df = pd.DataFrame(imputer_numericos.fit_transform(df), columns=columnas)

    return df
