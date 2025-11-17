import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Carga el archivo CSV "database_titanic.csv" en un DataFrame de pandas.
df = pd.read_csv("database_titanic.csv")

# Muestra un título y una descripción en la aplicación Streamlit.
st.write("""
# Mi primera aplicación interactiva
## Gráficos usando la base de datos del Titanic
""")

# Usando la notación "with" para crear una barra lateral en la aplicación Streamlit.
with st.sidebar:
    # Título para la sección de opciones en la barra lateral.
    st.write("# OPCIONES")
    
    # Crea un control deslizante (slider) que permite al usuario seleccionar un número de bins
    # en el rango de 0 a 10, con un valor predeterminado de 2.
    div = st.slider('Número de bins:', 0, 10, 2)
    
    # Muestra el valor actual del slider en la barra lateral.
    st.write("Bins=", div)

# Desplegamos un histograma con los datos del eje X
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].hist(df["Age"], bins=div)
ax[0].set_xlabel("Edad")
ax[0].set_ylabel("Frecuencia")
ax[0].set_title("Histograma de edades")

# Tomando datos para hombres y contando la cantidad
df_male = df[df["Sex"] == "male"]
cant_male = len(df_male)

# Tomando datos para mujeres y contando la cantidad
df_female = df[df["Sex"] == "female"]
cant_female = len(df_female)

ax[1].bar(["Masculino", "Femenino"], [cant_male, cant_female], color = "red")
ax[1].set_xlabel("Sexo")
ax[1].set_ylabel("Cantidad")
ax[1].set_title('Distribución de hombres y mujeres')

# Desplegamos el gráfico
st.pyplot(fig)

st.write("""
## Muestra de datos cargados
""")
# Graficamos una tabla
st.table(df.head())

# Normalizar Survived a 0/1 por seguridad
s = df["Survived"].copy()
if s.dtype == object:
    s = s.str.lower().map(lambda x: 1 if x in ("yes", "y", "si", "s", "1", "true", "t") else 0)
else:
    s = s.fillna(0).astype(int)
df = df.assign(Survived_norm=s)

# Agrupar y obtener valores por sexo
sexos = df.groupby("Sex")["Survived_norm"].sum()  # ej. index: ['female','male']

# Asegurar orden y nombres legibles
male_count = int(sexos.get("male", 0))
female_count = int(sexos.get("female", 0))
labels = ["Masculino", "Femenino"]
values = [male_count, female_count]

# Crear/usar ax[2] en una figura con 3 subplots (ejemplo)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))  # adapta tamaño según necesites

# (Aquí puedes mantener tus otros dos gráficos en ax[0] y ax[1])

# Grafico de supervivientes por sexo en ax[2]
ax[2].bar(labels, values, color="red")
ax[2].set_xlabel("Sexo")
ax[2].set_ylabel("Cantidad de supervivientes")
ax[2].set_title("Distribución de Supervivientes")
for i, v in enumerate(values):
    ax[2].text(i, v + max(1, int(0.01 * max(values))), str(v), ha='center')

