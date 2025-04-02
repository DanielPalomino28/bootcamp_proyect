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
src_datos_procesados = src_datos_brutos.replace("datos_unificados.csv", "datos_procesados_v3.csv") 

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


def guardar_df_procesado(df):
    # Guarda el DataFrame modificado en la misma ruta donde se encuentra el archivo original      
    df.to_csv(src_datos_procesados, index=False)
    print(f"DataFrame guardado en: {src_datos_procesados}")

def validar_rangos_originales(df, columnas_info):
    """
    Para cada columna en el diccionario (con nombre original),
    reemplaza con NaN los valores que no estén en el rango [min, max].
    """
    for col, info in columnas_info.items():
        if col in df.columns and type(col) != str:
            min_val = info["min"]
            max_val = info["max"]
            df[col] = df[col].apply(lambda x: x if pd.isnull(x) or (min_val <= x <= max_val) else np.nan)
    return df

def procesar_df(df):
    """
    Aplica las transformaciones avanzadas al DataFrame:
      0. Elimina columnas redundantes.
      1. Compara P6440 y P6450 y elimina la que tenga más valores nulos.
      2. Consolida la información de registro formal en 'empresa_formal' y la de subsidios en 'Recibe_Subsidio'.
      3. Recodifica respuestas 'no sabe/no responde' (valor 9) a NaN.
      4. Elimina filas con demasiadas celdas vacías.
      5. Consolida la información salarial en la columna 'Salario'.
      6. Valida rangos originales de las columnas y reemplaza valores fuera de rango por NaN.
      7. Imputa los valores nulos y realiza ajustes adicionales.
      8. Renombra columnas para facilitar el análisis.
      9. Guarda el DataFrame procesado en la misma ruta donde se encuentra el archivo original.
    """
    # 0. Eliminar columnas con demasiados campos vacios o no útiles.
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
    
    '''
    0.1 Diccionario extendido: 
        clave = nombre original, 
        valor = dict con nuevo nombre, 
        mínimo y máximo permitidos.
    '''
    columnas_info = {
        "P3046": {"nuevo_nombre": "Tiene_Contador", "min": 1, "max": 2},
        "P3363": {"nuevo_nombre": "Medio_Conseguido_Empleo", "min": 1, "max": 6},
        "P3364": {"nuevo_nombre": "Le_Descontaron_Retencion", "min": 1, "max": 2},
        "P6400": {"nuevo_nombre": "Misma_Empresa_Contratante", "min": 1, "max": 2},
        "P6450": {"nuevo_nombre": "Contrato_Verbal_Escrito", "min": 1, "max": 2},
        "P6460": {"nuevo_nombre": "Contrato_Termino", "min": 1, "max": 2},
        "P6920": {"nuevo_nombre": "Cotiza_Pension", "min": 1, "max": 3},
        "P6990": {"nuevo_nombre": "Afiliado_ARP", "min": 1, "max": 2},
        "P9450": {"nuevo_nombre": "Afiliado_Caja_Compensacion", "min": 1, "max": 2},
        "P1800": {"nuevo_nombre": "Tiene_Empleados", "min": 1, "max": 2},
        "P1881": {"nuevo_nombre": "Medio_Transporte_Trabajo", "min": 1, "max": 14},
        "P1882": {"nuevo_nombre": "Tiempo_Desplazamiento_Trabajo", "min": 0, "max": 300},
        "P3047": {"nuevo_nombre": "Quien_Decide_Horario", "min": 1, "max": 4},
        "P3048": {"nuevo_nombre": "Quien_Decide_Produccion", "min": 1, "max": 4},
        "P3049": {"nuevo_nombre": "Quien_Decide_Precio", "min": 1, "max": 4},
        "P3069": {"nuevo_nombre": "Total_Empleados_Empresa", "min": 1, "max": 10},
        "P6422": {"nuevo_nombre": "Conforme_Tipo_Contrato", "min": 1, "max": 2},
        "P6424S1": {"nuevo_nombre": "Vacaciones_Con_Sueldo", "min": 1, "max": 2},        
        "P6424S3": {"nuevo_nombre": "Derecho_Cesantia", "min": 1, "max": 2},
        "P6424S5": {"nuevo_nombre": "Licencia_Enfermedad_Pagada", "min": 1, "max": 2},
        "P6426": {"nuevo_nombre": "Tiempo_Trabajo_Empresa", "min": 0, "max": 720},
        "P6430": {"nuevo_nombre": "Tipo_Trabajo", "min": 1, "max": 9},
        "P6630S6": {"nuevo_nombre": "Pagos_Accidentes", "min": 1, "max": 2},
        "P6800": {"nuevo_nombre": "Horas_Semanales_Trabajo", "min": 1, "max": 126},
        "P6880": {"nuevo_nombre": "Lugar_Principal_Trabajo", "min": 1, "max": 11},
        "P6915": {"nuevo_nombre": "Cubrir_Costos_Enfermedad", "min": 1, "max": 12},
        "P6930": {"nuevo_nombre": "Fondo_Afiliado", "min": 1, "max": 4},
        "P6940": {"nuevo_nombre": "Quien_Paga_Pension", "min": 1, "max": 4},
        "P7020": {"nuevo_nombre": "Tuvo_Trabajo_Anterior", "min": 1, "max": 2},
        "P7130": {"nuevo_nombre": "Desea_Cambiar_Trabajo", "min": 1, "max": 2},
        "P7170S6": {"nuevo_nombre": "Satisfaccion_Jornada_Laboral", "min": 1, "max": 2},
        "P7240": {"nuevo_nombre": "Fuente_Ingresos_Sin_Trabajo", "min": 1, "max": 10},
        "P760": {"nuevo_nombre": "Meses_Sin_Empleo", "min": 0, "max": 99},
        "P9440": {"nuevo_nombre": "Consiguio_Empleo_Internet", "min": 1, "max": 2}
    }


    '''1. Comparar:
        -¿para realizar este trabajo, tiene usted algún tipo de contrato? (P6440)
        -¿el contrato es verbal o escrito? (P6450), 
        eliminando la columna que tenga más nulos.'''
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

    '''
    2.1 Consolidar información de registro formal en 'empresa_formal'
    Columnas para Registro formal: 
        La empresa o negocio en la que ... realiza su trabajo ¿está registrada ante la cámara de comercio? (¿tiene registro mercantil?) (P3066)
        La empresa, negocio o institución en la que trabaja está registrada o tiene Cámara de comercio(P3045S1)
        La empresa, negocio o institución en la que trabaja está registrada o tiene Rut (P3045S2)
        La empresa, negocio o institución en la que trabaja está registrada o tiene Personería Jurídica (P3045S3)
        La empresa o negocio en la que ... realiza su trabajo ¿está registrada ante la cámara de comercio? (¿tiene registro mercantil?) (P3065)
    '''

    registro_cols = ["P3045S1", "P3045S2", "P3045S3", "P3065", "P3066"]
    cols_existentes_reg = [col for col in registro_cols if col in df.columns]
    if cols_existentes_reg:
        #1 = afirmativo.
        df["empresa_formal"] = df[cols_existentes_reg].apply(
            lambda row: any(x == 1 for x in row if pd.notnull(x)), axis=1
        ).astype(int)
        print("Se consolidó la columna empresa_formal.")
        # Eliminar las columnas originales usadas para la consolidación.
        df.drop(columns=cols_existentes_reg, inplace=True)
    else:
        print("No se encontraron columnas para consolidar 'empresa_formal'.")

    '''
    2.2Consolidar información de subsidios en 'Recibe_Subsidio'
    Columnas para Subsidios: 
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

    '''
    2.3 Consolidar información de primas en 'Recibe_Prima'
        P6424S2: Representa la Prima_Navidad, es decir, indica si el trabajador recibe un bono o prima navideña.
        P6630S1: Representa la Prima_Servicios, que señala si el trabajador recibe un bono por servicios.
        P6630S2: Representa la Prima_Navidad_2, que puede ser una variante o complemento a la prima navideña.
        P6630S3: Representa la Prima_Vacaciones, que indica si el trabajador recibe un bono o prima durante las vacaciones.
    '''
    prima_cols = ["P6424S2", "P6630S1", "P6630S2", "P6630S3"]
    cols_existentes_prima = [col for col in prima_cols if col in df.columns]
    if cols_existentes_prima:
        # Se asume que 1 = afirmativo, por lo que si en alguna de estas columnas el valor es 1, se considera que recibe prima.
        df["Recibe_Prima"] = df[cols_existentes_prima].apply(
            lambda row: 1 if any(x == 1 for x in row if pd.notnull(x)) else 0, axis=1
        ).astype(int)
        print("Se consolidó la columna 'Recibe_Prima'.")
        # Eliminar las columnas originales usadas para la consolidación.
        df.drop(columns=cols_existentes_prima, inplace=True)
    else:
        print("No se encontraron columnas para consolidar 'Recibe_Prima'.")


    # 3. Recodificar respuestas "no sabe/no responde": valor 9 se reemplaza por NaN.
    # Dado que las respuestas son numéricas (1, 2, 9) o pueden estar vacías.
    for col in df.columns:
        if col not in ["P6430", "P1881","P6880"]: #No se tienen en cuenta las columnas P6430, P6880 y P1881, ya que no se requiere recodificación y son categóricas.
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].replace(9, np.nan)
            else:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').replace(9, np.nan)
                except Exception as e:
                    pass

    # 4. Eliminar filas con más de max_vacias celdas vacías (NaN, cadenas vacías o ".").
    max_vacias = 40 # Número máximo de celdas vacías permitidas por fila. (la mitad del total de columnas en este punto del df)
    empties = df.apply(lambda row: sum(pd.isnull(x) or (isinstance(x, str) and x.strip() in ["", "."]) for x in row), axis=1)
    filas_con_muchos_vacios = empties[empties > max_vacias].index
    if len(filas_con_muchos_vacios) > 0:
        df.drop(index=filas_con_muchos_vacios, inplace=True)
        print(f"Se eliminaron {len(filas_con_muchos_vacios)} registros con más de {max_vacias} celdas vacías.")
    else:
        print(f"No se encontraron registros con más de {max_vacias} celdas vacías.") 
    
    '''
    5. Consolidar la información salarial en la columna 'Salario'
        Ingresos laborales (INGLABO)
        ¿cuál fue la ganancia neta o los honorarios netos de ... Esa actividad, negocio, profesión o finca, el mes pasado? (P6750)
        Antes de descuentos ¿cuánto ganó ... El mes pasado en este empleo? (P6500)
        Nota: se toma la máxima de las tres columnas, ya que puede darse el caso en que entre encuestas la persona haya recibido un aumento
    ''' 
    if "INGLABO" in df.columns and "P6500" in df.columns and "P6750" in df.columns:
        df["Salario"] = df[["INGLABO", "P6500", "P6750"]].apply(
            lambda row: np.nan if row.isnull().all() or (row == "").all() or row.max() < 3 else row.max(), axis=1 #Se agrega condición (row.max() < 3) para evitar que se tomen valores como 1.0 o 2.0.
        )
        df.drop(columns=["INGLABO","P6500","P6750"], inplace=True)
        print("Se agregó la columna 'Salario' y se eliminaron las columnas 'INGLABO' y 'P6500'.") #Se eliminan las columnas INGLABO y P6500, ya que se consolidó la información en Salario_Mayor.         
    else:
        print("No se encontraron columnas INGLABO, P6750 y/o P6500 para calcular la columna Salario.")


    '''
    6. Validar rangos originales de las columnas y reemplazar valores fuera de rango por NaN.
        Se usa el diccionario 'columnas_info' para validar los rangos originales.
    '''
    df = validar_rangos_originales(df, columnas_info)
    print("Se validaron los rangos originales de las columnas y se reemplazaron los valores fuera de rango por NaN.")
    #reemplazar valores vacíos por NaN para porteriormente hacer la imputación de los nulos.
    df.replace(["", ".", " "], np.nan, inplace=True)

    '''
     # 7. Imputación de nulos y ajustes adicionales
        Rellenar celdas vacías en 
    '''
    print("Inició la imputación de los nulos.")

    '''
    7.1 Rellenar celdas vacías en la pregunta:
        En ese trabajo, ¿tiene empleados o personas que le ayudan en su negocio o actividad? (P1800) 
        con 2.0 que corresponde a "No tiene empleados".'
    '''
    if "P1800" in df.columns:
        df["P1800"].fillna(2.0, inplace=True)


    
    ''' 
    7.2 Rellenar celdas vacías en la pregunta:
        ¿cuántos? (P1800S1)
        con 0.0 que corresponde a "No tiene empleados" de P1800.        
    '''
    if "P1800S1" in df.columns:
        df["P1800S1"].fillna(0.0, inplace=True)


    # Si P7020 == 2.0, entonces P760 debe ser 0.0
    ''' 
    7.3 Rellenar celdas vacías en la pregunta:
        Antes del actual trabajo, ¿... Tuvo otro trabajo? (P7020)
        ¿cuántos meses estuvo sin empleo o trabajo ... Entre el trabajo actual y el anterior? (P760)        
    '''    
    if "P7020" in df.columns and "P760" in df.columns:
        df.loc[df["P7020"] == 2.0, "P760"] = 0.0
    

    # Si P1881 == 14.0, entonces P1882 debe ser 0.0
    ''' 
    7.4 Rellenar celdas vacías a partir de la pregunta:
        ¿Qué medio de transporte utiliza principalmente para desplazarse a su sitio de trabajo? (P1881) - 14 es "No se desplaza"
     Se realiza ajuste en:
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
    for col in num_cols:
        if col in columnas_info and columnas_info[col]["min"] == 1 and columnas_info[col]["max"] == 4:
            # Imputar con la moda si los valores están entre 1 y 4 según el diccionario
            moda = df[col].mode()
            if not moda.empty:
                df[col].fillna(moda.iloc[0], inplace=True)                
        else:
            # Imputar con la mediana en caso contrario
            mediana = df[col].median()
            df[col].fillna(mediana, inplace=True)

    # Para columnas de texto: rellenar con la moda
    # Seleccionar columnas categóricas
    cat_cols = df.select_dtypes(include=['object']).columns
    print(f"🔹 Columnas categóricas: {list(cat_cols)}")
    # Verificar si hay valores en la moda
    mode_values = df[cat_cols].mode()

    if not mode_values.empty:
        df[cat_cols] = df[cat_cols].fillna(mode_values.iloc[0])
        print(f"🔹 Campos categóricos vacíos rellenados con la moda.")
    else:
        print(f"⚠️ No se encontraron columnas tipo object en el DF para rellenar con la moda.")


    #Ordenar los datos de manera aleatoria para evitar sesgos en el análisis posterior.
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    
    #8. Recorre el diccionario y renombra las columnas existentes
    mapping = {col: info["nuevo_nombre"] for col, info in columnas_info.items() if col in df.columns}
    df.rename(columns=mapping, inplace=True)
    print(f"🔹 Se renombraron {len(mapping)} columnas.")

    # 🔹 Definir rangos válidos de valores
    columnas_info_filtrado = {
        "Medio_Transporte_Trabajo": {"min": 1, "max": 14},
        "Lugar_Principal_Trabajo": {"min": 1, "max": 8},
        "empresa_formal": {"min": 0, "max": 1},
        "Recibe_Subsidio": {"min": 0, "max": 1},
        "Recibe_Prima": {"min": 0, "max": 1},
        "Tipo_Trabajo": {"min": 1, "max": 9},
        "Salario": {"min": 0, "max": 10000000} 
    }

    # 🔹 Corregir valores fuera del rango en lugar de eliminarlos
    for col, info in columnas_info_filtrado.items():
        if col in df.select_dtypes(include=['float64', 'int64']).columns:  # Verificar que la columna no sea de tipo object
            fuera_de_rango = df[col].apply(lambda x: x < info["min"] or x > info["max"]).sum()
            df[col] = df[col].apply(lambda x: np.random.randint(info["min"], info["max"] + 1) if x < info["min"] or x > info["max"] else x)
            print(f"Columna {col}: {fuera_de_rango} registros corregidos por estar fuera del rango [{info['min']}, {info['max']}].")

    print("\n✅ Valores fuera de rango corregidos con valores aleatorios dentro del rango permitido.")

    #9. Guardar el DataFrame procesado en la misma ruta donde se encuentra el archivo original.
    df.to_csv(src_datos_procesados, index=False)
    print(f"🔹 DataFrame procesado guardado en: {src_datos_procesados}")

    return df

def generar_graficos():    
    # Mapa de calor de correlaciones
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd

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
        
        # Convertir el salario a millones para usarlo en los gráficos
        df["Salario_Millones"] = df["Salario"] / 1_000_000

        # Diagrama de dispersión: Salario vs. Horas Semanales de Trabajo, coloreando por Recibe_Subsidio
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="Horas_Semanales_Trabajo", y="Salario_Millones", hue="Recibe_Subsidio", palette="viridis", alpha=0.6)
        plt.title("Salario (millones) vs. Horas Semanales de Trabajo")
        plt.xlabel("Horas Semanales de Trabajo")
        plt.ylabel("Salario (millones)")
        plt.show()
        
        
        # Boxplot: Salario por Tipo de Trabajo (filtrando valores entre 1 y 9)
        df_filtered = df[df["Tipo_Trabajo"].between(1, 9)]  # Filtrar registros con valores entre 1 y 9
        # Calcular el máximo valor (redondeado hacia arriba) en millones
        max_salario_millones = int(np.ceil(df_filtered["Salario_Millones"].max()))
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_filtered, x="Tipo_Trabajo", y="Salario_Millones")
        plt.title("Distribución del Salario (millones) por Tipo de Trabajo")
        plt.xlabel("Tipo de Trabajo")
        plt.ylabel("Salario (millones)")
        plt.xticks(rotation=45)
        plt.yticks(np.arange(1, max_salario_millones + 1, 1))
        plt.show()
        
        # Barplot: Salario medio por Lugar Principal de Trabajo

        # Diccionario de mapeo para Lugar_Principal_Trabajo (nombres descriptivos)
        lugar_trabajo_dict = {
            1: "En esta vivienda",
            2: "En otras viviendas",
            3: "En kiosco - caseta",
            4: "En un vehículo",
            5: "De puerta en puerta",
            6: "Ambulante y estacionario",
            7: "Local fijo, oficina, fábrica, etc.",
            8: "En el campo o área rural, mar o río"
        }

        # Aplicar el mapeo a la columna Lugar_Principal_Trabajo
        df["Lugar_Principal_Trabajo"] = df["Lugar_Principal_Trabajo"].map(lugar_trabajo_dict)

        # Calcular el salario medio por lugar de trabajo
        salario_medio = df.groupby("Lugar_Principal_Trabajo")["Salario_Millones"].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(data=salario_medio, x="Lugar_Principal_Trabajo", y="Salario_Millones", palette="magma")
        plt.title("Salario Medio (millones) por Lugar Principal de Trabajo")
        plt.xlabel("Lugar Principal de Trabajo")
        plt.ylabel("Salario Medio (millones)")
        plt.xticks(rotation=45, ha="right")
        plt.show()
        
        # Boxplot: Salario por Medio de Transporte
        # Diccionario de mapeo para Medio_Transporte_Trabajo (abreviaciones o nombres breves)
        transporte_dict = {
            1: "Bus inter.",
            2: "Bus urbano",
            3: "A pie",
            4: "Metro",
            5: "Trans. articulado",
            6: "Taxi",
            7: "Transp. empresa",
            8: "Auto particular",
            9: "Lancha",
            10: "Caballo",
            11: "Moto",
            12: "Mototaxi",
            13: "Bicicleta",
            14: "No se desplaza"
        }

        # Aplicar el mapeo a la columna (suponiendo que los valores ya son numéricos)
        df["Medio_Transporte_Trabajo"] = df["Medio_Transporte_Trabajo"].map(transporte_dict)

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="Medio_Transporte_Trabajo", y="Salario_Millones", palette="Set2")
        plt.title("Distribución del Salario (millones) por Medio de Transporte")
        plt.xlabel("Medio de Transporte")
        plt.ylabel("Salario (millones)")
        plt.show()
        
        print("Gráficos generados. Revisa las visualizaciones para identificar relaciones importantes.")
    else:
        print(f"⚠️ No se encontró el archivo procesado en la ruta: {src_datos_procesados}.")





def main():
    df = cargar_datos(src_datos_brutos)
            # Se eliminan las columnas no útiles para el análisis.
    df.drop(columns=["PERIODO", "MES", "PER", "DIRECTORIO", "SECUENCIA_P", "ORDEN", "HOGAR", "REGIS", "AREA",
                    'CLASE', 'DPTO', 'FEX_C18', 'FT'], inplace=True, errors='ignore')
    print("▶ DataFrame cargado y columnas no útiles para análisis eliminadas.")
    while True:
        print("\nMenú de Análisis EDA:")
        print("1. Mostrar información general del DataFrame en bruto.")
        print("2. Revisar valores nulos.")
        print("3. Realizar procesamiento avanzado completo.")
        print("4. Generar gráficos de relación al DaraFrame procesado.")
        print("5. Salir.")
        
        choice = input("Seleccione una opción: ")
        
        if choice == "1":
            mostrar_dataframe(df)
        elif choice == "2":
            mostrar_nulos(df)
        elif choice == "3":
            df = procesar_df(df)
            print("Procesamiento avanzado completado en un solo paso. 💻") 
        elif choice == "4":
            if "Salario" not in df.columns:
                print("⚠️ Primero debe procesar el DataFrame para generar gráficos.")
            else:
                print("Generando gráficos exploratorios...")
                generar_graficos()            
        elif choice == "5":
            print("Gracias por usar el programa de análisis EDA. ¡Hasta luego! 👋")
            break
        else:
            print("Opción inválida. Intente de nuevo. 😕")

if __name__ == "__main__":
    main()
