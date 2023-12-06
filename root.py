import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Ruta del archivo de Excel
archivo_excel1 = '01_22.xlsx' # NO TIENE ASISITENCIA
archivo_excel2= '01_23.xlsx'
archivo_excel3 = '02_22.xlsx'


# Cargar el archivo de Excel en un DataFrame
dataframe1 = pd.read_excel(archivo_excel1, sheet_name='Sheet1')
dataframe2 = pd.read_excel(archivo_excel2, sheet_name='Reporte final')
dataframe3 = pd.read_excel(archivo_excel3, sheet_name='Notas')

# Eliminar entradas con nota cero en el examen final para cada DataFrame
dataframe1 = dataframe1[dataframe1['Examen final'] != 0]
dataframe2 = dataframe2[dataframe2['Parcial 4'] != 0]
dataframe3 = dataframe3[dataframe3['Examen Final'] != 0]



# Crear la variable dependiente 'y' para cada DataFrame
dataframe1['y'] = (dataframe1['Nota del curso'] >= 3).astype(int)
dataframe2['y'] = (dataframe2['Nota final nuevo esquema'] >= 3).astype(int)
dataframe3['y'] = (dataframe3['Nota Final'] >= 3).astype(int)

# Función para crear, entrenar y evaluar un modelo, devolviendo las matrices de confusión
def crear_evaluar_modelo(X, y, x_exclud, y_exclud):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Crear y entrenar el modelo de regresión logística
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    # Evaluar el modelo en los datos de entrenamiento y prueba
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    y_pred_exclud = modelo.predict(x_exclud)
    confusion_train = confusion_matrix(y_train, y_pred_train)
    confusion_test = confusion_matrix(y_test, y_pred_test)
    confusion_exclud = confusion_matrix(y_exclud, y_pred_exclud)

    return modelo, confusion_train, confusion_test, confusion_exclud

# Preparar variables independientes y dependiente para cada modelo
# Modelo 1: Combinar dataframe1 y dataframe2
X_modelo1 = pd.concat([dataframe1, dataframe2])[['Parcial 1', 'Parcial 2', 'Quiz 1', 'Quiz 2']]
y_modelo1 = pd.concat([dataframe1, dataframe2])['y']

# Modelo 2: Combinar dataframe2 y dataframe3
X_modelo2 = pd.concat([dataframe2, dataframe3])[['Parcial 1', 'Parcial 2', 'Quiz 1', 'Quiz 2']]
y_modelo2 = pd.concat([dataframe2, dataframe3])['y']

# Modelo 3: Combinar dataframe1 y dataframe3
X_modelo3 = pd.concat([dataframe1, dataframe3])[['Parcial 1', 'Parcial 2', 'Quiz 1', 'Quiz 2']]
y_modelo3 = pd.concat([dataframe1, dataframe3])['y']



#Datos dataframe1
X_dataframe1 = dataframe1[['Parcial 1', 'Parcial 2', 'Quiz 1', 'Quiz 2']]
y_dataframe1 = dataframe1['y']
#Datos dataframe2
X_dataframe2 = dataframe2[['Parcial 1', 'Parcial 2', 'Quiz 1', 'Quiz 2']]
y_dataframe2 = dataframe2['y']
#Datos dataframe3
X_dataframe3 = dataframe3[['Parcial 1', 'Parcial 2', 'Quiz 1', 'Quiz 2']]
y_dataframe3 = dataframe3['y']

# Crear y evaluar los tres modelos
modelo1, confusion_train1, confusion_test1, confusion_exclud1 = crear_evaluar_modelo(X_modelo1, y_modelo1, X_dataframe3, y_dataframe3)
modelo2, confusion_train2, confusion_test2, confusion_exclud2 = crear_evaluar_modelo(X_modelo2, y_modelo2, X_dataframe1, y_dataframe1)
modelo3, confusion_train3, confusion_test3, confusion_exclud3 = crear_evaluar_modelo(X_modelo3, y_modelo3, X_dataframe2, y_dataframe2)
#Graficador de matrix de confucion
def matrix_confucion(cm):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.show()

# Imprimir las matrices de confusión
print("Modelo 1 - Estimacion:", confusion_train1)
#matrix_confucion(confusion_train1)
#matrix_confucion(confusion_test1)
#matrix_confucion(confusion_exclud1)
print("Modelo 1 - Prueba:", confusion_test1)
print("Modelo 2 - Estimacion:", confusion_train2)
print("Modelo 2 - Prueba:", confusion_test2)
#matrix_confucion(confusion_train2)
#matrix_confucion(confusion_test2)
#matrix_confucion(confusion_exclud2)
print("Modelo 3 - Estimacion:", confusion_train3)
print("Modelo 3 - Prueba:", confusion_test3)
#matrix_confucion(confusion_train3)
#matrix_confucion(confusion_test3)
#matrix_confucion(confusion_exclud3)





# Combinar dataframe1 y dataframe3 para el Modelo 3
datos_modelo3 = pd.concat([dataframe1, dataframe3])

# Preparar variables independientes (X) y dependiente (y) para el modelo 3
X_modelo3 = datos_modelo3[['Parcial 1', 'Parcial 2', 'Quiz 1', 'Quiz 2']]
y_modelo3 = datos_modelo3['y']

# Crear y entrenar el modelo de regresión logística para el Modelo 3
modelo3 = LogisticRegression()
modelo3.fit(X_modelo3, y_modelo3)

# Crear un modelo nulo
X_nulo = np.ones((X_modelo3.shape[0], 1))  # Matriz de unos
modelo_nulo = LogisticRegression()
modelo_nulo.fit(X_nulo, y_modelo3)

# Calcular la devianza log-likelihood para ambos modelos
ll_full = -log_loss(y_modelo3, modelo3.predict_proba(X_modelo3), normalize=False)
ll_null = -log_loss(y_modelo3, modelo_nulo.predict_proba(X_nulo), normalize=False)

# Calcular la estadística de la prueba y el valor p
lr_stat = 2 * (ll_full - ll_null)
p_valor = stats.chi2.sf(lr_stat, df=X_modelo3.shape[1])  # Grados de libertad = número de predictores

print(f"Estadística de la prueba: {lr_stat}")
print(f"Valor p: {p_valor}")


# Establecer el umbral en 0.7 (70%)
umbral_asistencia = 0.7

# Convertir los porcentajes a valores binarios en cada DataFrame
dataframe2['Asistencia_Binaria'] = (dataframe2['Porcentaje asistencia magistral'] >= umbral_asistencia).astype(int)
dataframe3['Asistencia_Binaria'] = (dataframe3['Porcentaje asistencia magistral'] >= umbral_asistencia).astype(int)


# Dividir los DataFrames en subconjuntos según la asistencia

subconjunto1_df2 = dataframe2[dataframe2['Asistencia_Binaria'] == 0]
subconjunto2_df2 = dataframe2[dataframe2['Asistencia_Binaria'] == 1]

subconjunto1_df3 = dataframe3[dataframe3['Asistencia_Binaria'] == 0]
subconjunto2_df3 = dataframe3[dataframe3['Asistencia_Binaria'] == 1]

# Preparar variables independientes y dependiente para subconjunto1 de dataframe3
X_sub1_df3 = subconjunto1_df3[['Parcial 1', 'Parcial 2', 'Quiz 1', 'Quiz 2']]
y_sub1_df3 = subconjunto1_df3['y']

# Crear el modelo para los subconjunto de dataframe3
modelo_sub1_df3 = LogisticRegression()
modelo_sub1_df3.fit(X_sub1_df3, y_sub1_df3)

# Obtener los coeficientes para subconjunto3
coeficientes_sub1_df3 = modelo_sub1_df3.coef_

# Preparar variables independientes y dependiente para subconjunto2 de dataframe3
X_sub2_df3 = subconjunto2_df3[['Parcial 1', 'Parcial 2', 'Quiz 1', 'Quiz 2']]
y_sub2_df3 = subconjunto2_df3['y']

# Crear y entrenar el modelo para subconjunto2 de dataframe1
modelo_sub2_df3 = LogisticRegression()
modelo_sub2_df3.fit(X_sub2_df3, y_sub2_df3)

# Obtener los coeficientes para subconjunto2
coeficientes_sub2_df3 = modelo_sub2_df3.coef_

#repito Dataframe 2

X_sub1_df2 = subconjunto1_df2[['Parcial 1', 'Parcial 2', 'Quiz 1', 'Quiz 2']]
y_sub1_df2 = subconjunto1_df2['y']

modelo_sub1_df2 = LogisticRegression()
modelo_sub1_df2.fit(X_sub1_df2, y_sub1_df2)

coeficientes_sub1_df2 = modelo_sub1_df2.coef_

X_sub2_df2 = subconjunto2_df2[['Parcial 1', 'Parcial 2', 'Quiz 1', 'Quiz 2']]
y_sub2_df2 = subconjunto2_df2['y']

modelo_sub2_df2 = LogisticRegression()
modelo_sub2_df2.fit(X_sub2_df2, y_sub2_df2)

coeficientes_sub2_df2 = modelo_sub2_df2.coef_
print(coeficientes_sub1_df2,coeficientes_sub1_df3,coeficientes_sub2_df2,coeficientes_sub2_df3)


def calcular_errores_estandar(X, y):
    X = sm.add_constant(X)  # Añadir intercepto
    modelo = sm.Logit(y, X).fit_regularized(method='l1', alpha=1.0)  
    return modelo.bse[1:]  # Devolver errores estándar (excluyendo el intercepto)


# Calcular errores estándar para cada subconjunto
errores_estandar_sub1_df2 = calcular_errores_estandar(X_sub1_df2, y_sub1_df2)
errores_estandar_sub1_df3 = calcular_errores_estandar(X_sub1_df3, y_sub1_df3)
errores_estandar_sub2_df2 = calcular_errores_estandar(X_sub2_df2, y_sub2_df2)
errores_estandar_sub2_df3 = calcular_errores_estandar(X_sub2_df3, y_sub2_df3)

def prueba_wald(coeficientes1, errores1, coeficientes2, errores2):
    # Asegurarse de que todos los inputs sean arrays de numpy y tengan la misma longitud
    coeficientes1 = np.array(coeficientes1).flatten()
    errores1 = np.array(errores1).flatten()
    coeficientes2 = np.array(coeficientes2).flatten()
    errores2 = np.array(errores2).flatten()

    # Verificar si las longitudes son iguales
    if len(coeficientes1) != len(coeficientes2) or len(errores1) != len(errores2):
        raise ValueError("Las longitudes de los coeficientes o errores estándar no coinciden.")

    # Calcular diferencias y varianzas
    diferencias = coeficientes1 - coeficientes2
    varianzas = errores1*2 + errores2*2

    # Calcular z-scores y p-values
    z_scores = diferencias / np.sqrt(varianzas)
    p_values = stats.norm.sf(abs(z_scores)) * 2  # Prueba de dos colas

    return p_values


# Realizar pruebas de Wald para comparar los coeficientes entre subconjuntos
p_values_df2 = prueba_wald(coeficientes_sub1_df2, errores_estandar_sub1_df2, coeficientes_sub2_df2, errores_estandar_sub2_df2)
p_values_df3 = prueba_wald(coeficientes_sub1_df3, errores_estandar_sub1_df3, coeficientes_sub2_df3, errores_estandar_sub2_df3)

print("P-values DataFrame 2:", p_values_df2)
print("P-values DataFrame 3:", p_values_df3)


def probar_hipotesis(X, y, X_restringido):
    # Modelo completo
    modelo_completo = LogisticRegression()
    modelo_completo.fit(X, y)
    ll_completo = -log_loss(y, modelo_completo.predict_proba(X), normalize=False)

    # Modelo restringido
    modelo_restringido = LogisticRegression()
    modelo_restringido.fit(X_restringido, y)
    ll_restringido = -log_loss(y, modelo_restringido.predict_proba(X_restringido), normalize=False)

    # Calcular la estadística de la prueba y el valor p
    lr_stat = 2 * (ll_completo - ll_restringido)
    p_valor = stats.chi2.sf(lr_stat, df=1)  # 1 grado de libertad (asistencia)

    return lr_stat, p_valor

# DataFrame 2
X_df2 = dataframe2[['Parcial 1', 'Parcial 2', 'Quiz 1', 'Quiz 2', 'Asistencia_Binaria']]
X_restringido_df2 = dataframe2[['Parcial 1', 'Parcial 2', 'Quiz 1', 'Quiz 2']]
y_df2 = dataframe2['y']
lr_stat_df2, p_valor_df2 = probar_hipotesis(X_df2, y_df2, X_restringido_df2)

# DataFrame 3
X_df3 = dataframe3[['Parcial 1', 'Parcial 2', 'Quiz 1', 'Quiz 2', 'Asistencia_Binaria']]
X_restringido_df3 = dataframe3[['Parcial 1', 'Parcial 2', 'Quiz 1', 'Quiz 2']]
y_df3 = dataframe3['y']
lr_stat_df3, p_valor_df3 = probar_hipotesis(X_df3, y_df3, X_restringido_df3)

print(f"DataFrame 2: Estadística LR = {lr_stat_df2}, P-valor = {p_valor_df2}")
print(f"DataFrame 3: Estadística LR = {lr_stat_df3}, P-valor = {p_valor_df3}")