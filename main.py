#Colab
    #from google.colab import drive
    #drive.mount('/content/drive')
#Local



# Importación de librerías
import pandas as pd

# Carga del archivo CSV
df_complete = pd.read_csv(r"C:\Users\danie\Documents\Bootcamp\data.csv")

# Creación del dataframe
df = pd.DataFrame(df_complete)

# Obtiene la cantidad de nulos por columna
nulos_por_columna = df.isnull().sum()


print(nulos_por_columna.head)