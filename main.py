#Colab
    #from google.colab import drive
    #drive.mount('/content/drive')
#Local



# Importación de librerías
import pandas as pd

# Carga del archivo CSV
df_complete = pd.read_csv("/content/drive/MyDrive/DataSets For Proyect/Copia de datos_unificados.csv")

# Creación del dataframe
df = pd.DataFrame(df_complete)

# Obtiene la cantidad de nulos por columna
nulos_por_columna = df.df_complete().sum()


print(nulos_por_columna.head)