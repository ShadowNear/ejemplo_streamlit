import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Carga CSV
df = pd.read_csv(r"database_titanic.csv")

st.write("# Mi primera aplicación interactiva\n## Gráficos usando la base de datos del Titanic")

with st.sidebar:
    st.write("# OPCIONES")
    bins = st.slider("Número de bins:", 1, 20, 8)
    st.write("Bins=", bins)

# Normalizar Survived a 0/1
s = df["Survived"].copy()
if s.dtype == object:
    s = s.str.lower().map(lambda x: 1 if x in ("yes", "y", "si", "s", "1", "true", "t") else 0)
else:
    s = s.fillna(0).astype(int)
df = df.assign(Survived_norm=s)

# Crear figura con 3 subplots (1 fila x 3 columnas)
fig, ax = plt.subplots(1, 3, figsize=(18, 4))

# 1) Histograma de Age
ax[0].hist(df["Age"].dropna(), bins=bins, color="skyblue", edgecolor="k")
ax[0].set_xlabel("Edad")
ax[0].set_ylabel("Frecuencia")
ax[0].set_title("Histograma de edades")

# 2) Conteo total por sexo
counts = df["Sex"].value_counts()
labels_counts = ["Masculino" if s == "male" else "Femenino" if s == "female" else s for s in counts.index]
ax[1].bar(labels_counts, counts.values, color=["#d9534f", "#5bc0de"][:len(counts)])
ax[1].set_xlabel("Sexo")
ax[1].set_ylabel("Cantidad")
ax[1].set_title("Distribución de hombres y mujeres")
for i, v in enumerate(counts.values):
    ax[1].text(i, v + max(1, int(0.01 * max(counts.values))), int(v), ha='center')

# 3) Supervivientes por sexo (usar valores sumados)
surv_by_sex = df.groupby("Sex")["Survived_norm"].sum()
male_count = int(surv_by_sex.get("male", 0))
female_count = int(surv_by_sex.get("female", 0))
ax[2].bar(["Masculino", "Femenino"], [male_count, female_count], color=["#4CAF50", "#FF9800"])
ax[2].set_xlabel("Sexo")
ax[2].set_ylabel("Cantidad de supervivientes")
ax[2].set_title("Supervivientes por sexo")
for i, v in enumerate([male_count, female_count]):
    ax[2].text(i, v + max(1, int(0.01 * max(male_count, female_count))), str(v), ha='center')

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.write("## Muestra de datos cargados")
st.table(df.head())



