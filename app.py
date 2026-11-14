import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

st.set_page_config(page_title="California Housing", layout="wide")

st.title("Análisis Interactivo de Viviendas en California")
st.markdown("Datos del censo de 1990 en California (sklearn.datasets).")


# ---------------------------------------------------------------
# FASE 1: Carga y preparación de datos con Pandas
# ---------------------------------------------------------------
@st.cache_data  # guarda los datos en memoria para no descargarlos en cada interacción
def cargar_datos():
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df["MedHouseVal"] = california.target
    return df


df_california = cargar_datos()

st.header("1. Carga y preparación de datos")

st.subheader("Primeras 5 filas")
st.dataframe(df_california.head())

col1, col2 = st.columns(2)

with col1:
    st.subheader("Tipo de dato de cada columna")
    st.write(df_california.dtypes.astype(str).rename("Tipo"))

with col2:
    st.subheader("Valores faltantes por columna")
    faltantes = df_california.isnull().sum()
    st.write(faltantes.rename("Faltantes"))
    if faltantes.sum() == 0:
        st.success("El dataset no tiene valores faltantes.")
    else:
        st.warning(f"El dataset tiene {faltantes.sum()} valores faltantes.")

st.write(f"Cantidad total de registros: {len(df_california)}")


# ---------------------------------------------------------------
# FASE 2: Análisis descriptivo interactivo (sidebar con filtros)
# ---------------------------------------------------------------
st.sidebar.markdown("## Filtros")
st.sidebar.markdown(
    "Usá estos controles para filtrar los datos. "
    "Todas las tablas y gráficos se actualizan automáticamente."
)

# Filtro por edad mediana de la casa (slider de rango)
edad_min = float(df_california["HouseAge"].min())
edad_max = float(df_california["HouseAge"].max())

rango_edad = st.sidebar.slider(
    "Edad mediana de la casa (HouseAge)",
    min_value=edad_min,
    max_value=edad_max,
    value=(edad_min, edad_max),
    step=1.0,
)

df_filtrado = df_california[
    (df_california["HouseAge"] >= rango_edad[0])
    & (df_california["HouseAge"] <= rango_edad[1])
]

# Filtro de vecindario: checkbox para activarlo + número para la latitud mínima
lat_min = float(df_california["Latitude"].min())
lat_max = float(df_california["Latitude"].max())

usar_filtro_latitud = st.sidebar.checkbox("Filtrar por latitud mínima")

if usar_filtro_latitud:
    latitud_minima = st.sidebar.number_input(
        "Latitud mínima",
        min_value=lat_min,
        max_value=lat_max,
        value=lat_min,
        step=0.5,
    )
    df_filtrado = df_filtrado[df_filtrado["Latitude"] >= latitud_minima]

# Resumen descriptivo
st.header("2. Resumen descriptivo (datos filtrados)")

if df_filtrado.empty:
    st.error("No hay datos con los filtros seleccionados. Probá con otros valores.")
    st.stop()

mediana_valor = df_filtrado["MedHouseVal"].median()
rango_valor = df_filtrado["MedHouseVal"].max() - df_filtrado["MedHouseVal"].min()

c1, c2, c3 = st.columns(3)
c1.metric("Registros filtrados", len(df_filtrado))
c2.metric("Mediana de MedHouseVal", f"{mediana_valor:.2f}")
c3.metric("Rango de MedHouseVal (máx - mín)", f"{rango_valor:.2f}")

st.caption("MedHouseVal está expresado en cientos de miles de dólares.")


# ---------------------------------------------------------------
# FASE 3: Visualización dinámica
# ---------------------------------------------------------------
st.header("3. Visualizaciones")

col_a, col_b = st.columns(2)

# Histograma del target
with col_a:
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(df_filtrado["MedHouseVal"], bins=30, color="skyblue", edgecolor="black")
    ax_hist.set_title("Distribución del valor mediano de la vivienda")
    ax_hist.set_xlabel("MedHouseVal (cientos de miles de USD)")
    ax_hist.set_ylabel("Frecuencia")
    st.pyplot(fig_hist)

# Dispersión ingresos vs. valor de la vivienda
with col_b:
    fig_disp, ax_disp = plt.subplots()
    ax_disp.scatter(df_filtrado["MedInc"], df_filtrado["MedHouseVal"], alpha=0.3, s=5)
    ax_disp.set_title("Ingreso mediano vs. valor mediano de la vivienda")
    ax_disp.set_xlabel("MedInc (decenas de miles de USD)")
    ax_disp.set_ylabel("MedHouseVal (cientos de miles de USD)")
    st.pyplot(fig_disp)

# Opcional: mapa geográfico
st.subheader("Mapa de los grupos de bloques filtrados")
df_mapa = df_filtrado.rename(columns={"Latitude": "lat", "Longitude": "lon"})
st.map(df_mapa[["lat", "lon"]])
