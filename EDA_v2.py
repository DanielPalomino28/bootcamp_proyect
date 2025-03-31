import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import os
import requests
import re

# URL del archivo CSV (ruta local en este ejemplo)
src_datos_brutos = r"Z:\datos_unificados.csv"
src_datos_procesados = src_datos_brutos.replace("datos_unificados.csv", "datos_procesados.csv") 

def cargar_datos(source, bloques=450000):
    # Carga el dataset en bloques de 450,000 registros
    chunks = pd.read_csv(source, chunksize=bloques, low_memory=True)
    # Combina los bloques en un solo DataFrame
    return pd.concat(chunks, ignore_index=True)

def mostrar_dataframe(df):
    # Muestra información general del DataFrame
    print("\nInformación del DataFrame:")
    print(df.info())
    
    print("\nResumen estadístico:")
    print(df.describe(include='all'))

def mostrar_nulos(df):
    # Muestra la cantidad de valores nulos por columna
    print("\nValores nulos por columna:")
    print(df.isnull().sum().sort_values(ascending=False))

def graficar_dist_num(df):
    # Genera gráficos de distribución para variables numéricas
    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    df[columnas_numericas].hist(figsize=(15, 10), bins=30)
    plt.show()

def graficar_dist_cat(df):
    # Genera gráficos de barras para variables categóricas
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        plt.figure(figsize=(10, 5))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'Distribución de {col}')
        plt.show()

def view_column_values(df):
    # Muestra los primeros 20 valores de una columna específica
    col = input("Ingrese el nombre de la columna a visualizar: ")
    if col in df.columns:
        print(df[col].head(20))
    else:
        print("Columna no encontrada.")


def guardar_df_procesado(df, src_datos):
    # Guarda el DataFrame modificado en la misma ruta donde se encuentra el archivo original      
    df.to_csv(src_datos, index=False)
    print(f"DataFrame guardado en: {src_datos}")

def procesar_df(df):
    """
    Aplica las transformaciones avanzadas al DataFrame según lo solicitado:
      0. Elimina columnas redundantes definidas en la lista (para no tener información ya optimizada).
      1. Compara las columnas P6440 y P6450 y elimina la que tenga más valores nulos.
      2. Consolida información de registro formal y de subsidios y elimina las columnas originales.
      3. Recodifica las respuestas "no sabe/no responde" (valor 9) a NaN en columnas numéricas y agrega
         una columna con el conteo de celdas vacías.
      4. Elimina filas con más de 100 celdas vacías.
    """
    # 0. Eliminar columnas con demasiados campos vacios o 
    drop_columns_initial = [
        "P6410", "P6420S2", "P6430S1", "P3364S1", "P6590S1", "P6600", "P6600S1", "P6610S1", "P6610", "P6585S3A1", 
        "P6585S3A2", "P6545S1", "P6545S2", "P6765S1", "P3051", "P3051S1", "P3055S1", "P3056", "P3057","P6760", 
        "P3058S1", "P3058S2", "P3058S3", "P3058S4", "P3058S5", "P3059", "P3061", "P3062S1", "P3062S2", "P3062S3", 
        "P3062S4", "P3062S5","P3062S6", "P3062S7", "P3062S8", "P3062S9", "P3063", "P3063S1", "P3064", "P3064S1",
        "P3067", "P3067S1", "P3067S2", "P6775", "P3073", "P550", "P6780S1", "P6810", "P6810S1", "P6830", 
        "P6830S1", "P3366", "P6880S1", "P7028S1", "P7045", "P7050", "P7070", "P7075", "P7077", "P7100", 
        "P7110", "P7120", "P7140S1", "P7140S2", "P7140S3", "P7140S4","P7140S5", "P7140S6", "P7140S7", "P7140S8", 
        "P7140S9", "P7150", "P7160", "P6460S1", "P6510S1", "P6510S2", "P6620S1", "P6585S1A1", "P6585S1A2", "P6585S2A1", 
        "P6585S2A2", "P6585S4A1", "P6585S4A2", "P6580S1", "P6580S2", "P6630S2A1", "P6630S3A1", "P6630S4A1", "P6630S6A1","P6640S1", 
        "P1800S1", "P1801S1", "P1801S2", "P1801S3", "P6780", "P1879", "P1805", "P6850", "P6915S1", "P6960", 
        "P7028", "P1880", "P1880S1", "P7040", "P7090", "P7170S1", "P7170S5", "P7180", "P514", "P515", 
        "OCI", "OFICIO_C8","P1802", "P3044S2", "P3052", "P3052S1", "P3053", "P3054", "P3054S1", "P3055",
        "P3068", "P3365", "P3365S1", "P6510", "P6545", "P6580", "P6590", "P6620", "P6630S1A1", "P6630S4", 
        "P6640", "P6765", "P6790", "P7026", "RAMA2D_R4", "RAMA4D_R4"
    ]
    df.drop(columns=[col for col in drop_columns_initial if col in df.columns], inplace=True)
    print(f"Se eliminaron {len(drop_columns_initial)} columnas redundantes al inicio.")
    

    '''1. Comparar ¿para realizar este trabajo, tiene usted algún tipo de contrato? (P6440)
         y ¿el contrato es verbal o escrito? (P6450), eliminando la columna que tenga más nulos.'''
    if "P6440" in df.columns and "P6450" in df.columns:
        nulos_p6440 = df["P6440"].isnull().sum() + (df["P6440"] == "").sum()
        nulos_p6450 = df["P6450"].isnull().sum() + (df["P6450"] == "").sum()
        if nulos_p6440 > nulos_p6450:
            df.drop(columns=["P6440"], inplace=True)
            print("Se eliminó la columna P6440 (más nulos que P6450).")
        else:
            df.drop(columns=["P6450"], inplace=True)
            print("Se eliminó la columna P6450 (más nulos que P6440).")
    else:
        print("No se encontraron ambas columnas P6440 y P6450 para comparar.")

    # 2. Consolidar información de registro formal y de subsidios.
        '''Columnas para Registro formal: 
            La empresa o negocio en la que ... realiza su trabajo ¿está registrada ante la cámara de comercio? (¿tiene registro mercantil?) (P3066)
            La empresa, negocio o institución en la que ….. trabaja ¿está registrada o tiene: (P3045S1)
            La empresa, negocio o institución en la que ….. trabaja ¿está registrada o tiene: (P3045S2)
            La empresa, negocio o institución en la que ….. trabaja ¿está registrada o tiene: (P3045S3)
            La empresa o negocio en la que ... realiza su trabajo ¿está registrada ante la cámara de comercio? (¿tiene registro mercantil?) (P3065)'''

    registro_cols = ["P3045S1", "P3045S2", "P3045S3", "P3065", "P3066"]
    cols_existentes_reg = [col for col in registro_cols if col in df.columns]
    if cols_existentes_reg:
        # Se asume 1 = afirmativo.
        df["Empresa_Registrada"] = df[cols_existentes_reg].apply(
            lambda row: any(x == 1 for x in row if pd.notnull(x)), axis=1
        ).astype(int)
        print("Se consolidó la columna Empresa_Registrada.")
        # Eliminar las columnas originales usadas para la consolidación.
        df.drop(columns=cols_existentes_reg, inplace=True)
    else:
        print("No se encontraron columnas para consolidar Empresa_Registrada.")

    '''Columnas para Subsidios: 
            Auxilio o subsidio de alimentación? (P6585S1)
            Auxilio o subsidio de transporte? (P6585S2)
            Subsidio familiar? (P6585S3)
            Subsidio educativo? (P6585S4)
    '''
    subsidio_cols = ["P6585S1", "P6585S2", "P6585S3", "P6585S4"]
    cols_existentes_sub = [col for col in subsidio_cols if col in df.columns]
    if cols_existentes_sub:
        df["Recibe_Subsidio"] = df[cols_existentes_sub].apply(
            lambda row: any(x == 1 for x in row if pd.notnull(x)), axis=1
        ).astype(int)
        print("Se consolidó la columna Recibe_Subsidio.")
        # Eliminar las columnas originales usadas para consolidar subsidios.
        df.drop(columns=cols_existentes_sub, inplace=True)
    else:
        print("No se encontraron columnas para consolidar Recibe_Subsidio.")

    

    # 3. Recodificar respuestas "no sabe/no responde": valor 9 se reemplaza por NaN.
    # Dado que las respuestas son numéricas (1, 2, 9) o pueden estar vacías.
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace(9, np.nan)
        else:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').replace(9, np.nan)
            except Exception as e:
                pass

    # 4. Eliminar filas con más de max_vacias celdas vacías (NaN, cadenas vacías o ".").
    max_vacias = 60
    empties = df.apply(lambda row: sum(pd.isnull(x) or (isinstance(x, str) and x.strip() in ["", "."]) for x in row), axis=1)
    filas_con_muchos_vacios = empties[empties > max_vacias].index
    if len(filas_con_muchos_vacios) > 0:
        df.drop(index=filas_con_muchos_vacios, inplace=True)
        print(f"Se eliminaron {len(filas_con_muchos_vacios)} registros con más de {max_vacias} celdas vacías.")
    else:
        print(f"No se encontraron registros con más de {max_vacias} celdas vacías.") 

    # Agregar columna con el salario mayor entre las columnas siguientes columnas (en caso de no exista alguno de los dos registros, se crea un NaN.)
    '''Ingresos laborales (INGLABO)
        ¿cuál fue la ganancia neta o los honorarios netos de ... Esa actividad, negocio, profesión o finca, el mes pasado? (P6750)
        Antes de descuentos ¿cuánto ganó ... El mes pasado en este empleo? (P6500)
    ''' 
    if "INGLABO" in df.columns and "P6500" in df.columns and "P6750" in df.columns:
        df["Salario"] = df[["INGLABO", "P6500","P6750"]].apply(
            lambda row: np.nan if row.isnull().all() or (row == "").all() else row.max(), axis=1
        )
        df.drop(columns=["INGLABO","P6500","P6750"], inplace=True)
        print("Se agregó la columna 'Salario' y se eliminaron las columnas 'INGLABO' y 'P6500'.") #Se eliminan las columnas INGLABO y P6500, ya que se consolidó la información en Salario_Mayor.         
    else:
        print("No se encontraron columnas INGLABO, P6750 o P6500 para calcular la columna Salario.")


    #reemplazar valores vacíos por NaN para porteriormente hacer la imputación de los nulos.
    df.replace(["", ".", " "], np.nan, inplace=True)

    print("Inició la imputación de los nulos.")

    ''' Rellenar celdas vacías en 
        En ese trabajo, ¿tiene empleados o personas que le ayudan en su negocio o actividad? (P1800) con 2.0 que corresponde a "No tiene empleados".'
    '''
    if "P1800" in df.columns:
        df["P1800"].fillna(2.0, inplace=True)


    
    ''' Rellenar celdas vacías en ¿cuántos? (P1800S1)
        con 0.0 que corresponde a "No tiene empleados" de P1800.        
    '''
    if "P1800S1" in df.columns:
        df["P1800S1"].fillna(0.0, inplace=True)


    # Si P7020 == 2.0, entonces P760 debe ser 0.0
    ''' 
        Antes del actual trabajo, ¿... Tuvo otro trabajo? (P7020)
        ¿cuántos meses estuvo sin empleo o trabajo ... Entre el trabajo actual y el anterior? (P760)        
    '''    
    if "P7020" in df.columns and "P760" in df.columns:
        df.loc[df["P7020"] == 2.0, "P760"] = 0.0
    

    # Si P1881 == 14.0, entonces P1882 debe ser 0.0
    ''' 
    ¿Qué medio de transporte utiliza principalmente para desplazarse a su sitio de trabajo? (P1881) - 14 es "No se desplaza"
    ¿Cuánto tiempo se demora regularmente ... en su desplazamiento hacia el trabajo? (Incluya tiempo de espera del medio de transporte) (P1882)
    '''    
    if "P1881" in df.columns and "P1882" in df.columns:
        df.loc[df["P1881"] == 14.0, "P1882"] = 0.0

    # Diccionario con columna: valor a reemplazar por NaN
    valores_a_na = {
        "P3046": 9,
        "P3363": 9,
        "P3364": 9,
        "P6400": 9,
        "P6450": 3,
        "P6460": 3,
        "P6920": 3,
        "P6990": 9,
        "P9450": 9
    }

    # Recorre el diccionario y reemplaza el valor especificado por NaN en cada columna, ya que corresponde a "No sabe/no responde".
    for col, valor in valores_a_na.items():
        if col in df.columns:
            count_reemplazos = (df[col] == valor).sum()
            df[col] = df[col].replace(valor, np.nan)
            print(f"Columna {col}: {count_reemplazos} celdas cambiadas a NaN.")


    '''Rellenar los campos vacíos'''
    # Para columnas numéricas: rellenar con la mediana
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    print(f"🔹 Campos numéricos vacíos rellenados con la mediana.")

    # Para columnas categóricas: rellenar con la moda
    # Seleccionar columnas categóricas
    cat_cols = df.select_dtypes(include=['object']).columns
    print(f"🔹 Columnas categóricas: {cat_cols}")
    # Verificar si hay valores en la moda
    mode_values = df[cat_cols].mode()

    if not mode_values.empty:
        df[cat_cols] = df[cat_cols].fillna(mode_values.iloc[0])
        print(f"🔹 Campos categóricos vacíos rellenados con la moda.")
    else:
        print(f"⚠️ No se pudo calcular la moda para las columnas categóricas.")


    #Ordenar los datos de manera aleatoria para evitar sesgos en el análisis posterior.
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Renombrar columnas luego de la limpieza y procesamiento
    # Se renombra las columnas para facilitar la lectura y el análisis posterior.
    df.rename(columns={
        "P3046": "Tiene_Contador",
        "P3363": "Medio_Conseguido_Empleo",
        "P3364": "Le_Descontaron_Retencion",
        "P6400": "Misma_Empresa_Contratante",
        "P6450": "Contrato_Verbal_Escrito",
        "P6460": "Contrato_Termino",
        "P6920": "Cotiza_Pension",
        "P6990": "Afiliado_ARP",
        "P9450": "Afiliado_Caja_Compensacion",
        "P1800": "Tiene_Empleados",
        "P1881": "Medio_Transporte_Trabajo",
        "P1882": "Tiempo_Desplazamiento_Trabajo",
        "P3047": "Quien_Decide_Horario",
        "P3048": "Quien_Decide_Produccion",
        "P3049": "Quien_Decide_Precio",
        "P3069": "Total_Empleados_Empresa",
        "P6422": "Conforme_Tipo_Contrato",
        "P6424S1": "Vacaciones_Con_Sueldo",
        "P6424S2": "Prima_Navidad",
        "P6424S3": "Derecho_Cesantia",
        "P6424S5": "Licencia_Enfermedad_Pagada",
        "P6426": "Tiempo_Trabajo_Empresa",
        "P6430": "Tipo_Trabajo",
        "P6630S1": "Prima_Servicios",
        "P6630S2": "Prima_Navidad_2",
        "P6630S3": "Prima_Vacaciones",
        "P6630S6": "Pagos_Accidentes",
        "P6800": "Horas_Semanales_Trabajo",
        "P6880": "Lugar_Principal_Trabajo",
        "P6915": "Cubrir_Costos_Enfermedad",
        "P6930": "Fondo_Afiliado",
        "P6940": "Quien_Paga_Pension",
        "P7020": "Tuvo_Trabajo_Anterior",
        "P7130": "Desea_Cambiar_Trabajo",
        "P7170S6": "Satisfaccion_Jornada_Laboral",
        "P7240": "Fuente_Ingresos_Sin_Trabajo",
        "P760": "Meses_Sin_Empleo",
        "P9440": "Consiguio_Empleo_Internet"
    }, inplace=True)

    return df

def generar_graficos():    
    # Mapa de calor de correlaciones
    import seaborn as sns
    import matplotlib.pyplot as plt
    if os.path.exists(src_datos_procesados):
        df = pd.read_csv(src_datos_procesados)
        """
        Ejecuta todos los gráficos exploratorios sugeridos usando el DataFrame procesado.
        src_datos_procesados: ruta donde se guardó el DataFrame procesado (se puede usar para titulación o referencia).
        """
        print("Generando gráficos exploratorios con el DataFrame procesado...")
    
        # Mapa de calor de correlaciones filtrado a las 15 variables con mayor correlación promedio absoluta
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = df[num_cols].corr()
        mean_corr = corr_matrix.abs().mean().sort_values(ascending=False)
        top15_cols = mean_corr.head(15).index
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix.loc[top15_cols, top15_cols], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Mapa de Calor de Correlaciones (Top 15 variables con mayor correlación promedio)")
        plt.show()
        
        # Diagrama de dispersión: Salario vs. Horas Semanales de Trabajo, coloreando por Recibe_Subsidio
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="Horas_Semanales_Trabajo", y="Salario", hue="Recibe_Subsidio", palette="viridis", alpha=0.6)
        plt.title("Salario vs. Horas Semanales de Trabajo")
        plt.xlabel("Horas Semanales de Trabajo")
        plt.ylabel("Salario")
        plt.show()
        
        # Boxplot: Salario por Tipo de Trabajo
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="Tipo_Trabajo", y="Salario")
        plt.title("Distribución del Salario por Tipo de Trabajo")
        plt.xlabel("Tipo de Trabajo")
        plt.ylabel("Salario")
        plt.xticks(rotation=45)
        plt.show()
        
        # Barplot: Salario medio por Lugar Principal de Trabajo
        salario_medio = df.groupby("Lugar_Principal_Trabajo")["Salario"].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(data=salario_medio, x="Lugar_Principal_Trabajo", y="Salario", palette="magma")
        plt.title("Salario Medio por Lugar Principal de Trabajo")
        plt.xlabel("Lugar Principal de Trabajo")
        plt.ylabel("Salario Medio")
        plt.xticks(rotation=45)
        plt.show()
        
        # Boxplot: Salario por Medio de Transporte
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="Medio_Transporte_Trabajo", y="Salario", palette="Set2")
        plt.title("Distribución del Salario por Medio de Transporte")
        plt.xlabel("Medio de Transporte")
        plt.ylabel("Salario")
        plt.show()
    
        print("Gráficos generados. Revisa las visualizaciones para identificar relaciones importantes.")
    else:
        print(f"⚠️ No se encontró el archivo procesado en la ruta: {src_datos_procesados}.")




def main():
    while True:        
        if not os.path.exists(src_datos_procesados):
            df = cargar_datos(src_datos_brutos)
            # Se eliminan las columnas no útiles para el análisis.
            df.drop(columns=["PERIODO", "MES", "PER", "DIRECTORIO", "SECUENCIA_P", "ORDEN", "HOGAR", "REGIS", "AREA",
                            'CLASE', 'DPTO', 'FEX_C18', 'FT'], inplace=True, errors='ignore')
            print("▶ DataFrame cargado y columnas no útiles para análisis eliminadas.")

        print("\nMenú de Análisis EDA:")
        print("1. Mostrar información general del DataFrame en bruto.")
        print("2. Revisar valores nulos.")
        print("3. Realizar procesamiento avanzado completo.")
        print("4. Guardar DataFrame actualizado.")
        print("5. Generar gráficos de relación al DaraFrame procesado.")
        print("6. Salir.")
        
        choice = input("Seleccione una opción: ")
        
        if choice == "1":
            mostrar_dataframe(df)
        elif choice == "2":
            mostrar_nulos(df)
        elif choice == "3":
            df = procesar_df(df)
            print("Procesamiento avanzado completado en un solo paso. 💻")            
        elif choice == "4":
            guardar_df_procesado(df, src_datos_brutos)
        elif choice == "5":
            generar_graficos()
        elif choice == "6":
            print("Gracias por usar el programa de análisis EDA. ¡Hasta luego! 👋")
            break
        else:
            print("Opción inválida. Intente de nuevo. 😕")

if __name__ == "__main__":
    main()
