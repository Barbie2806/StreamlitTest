### Importe de librerias

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pygwalker as pyg
import streamlit.components.v1 as components
from io import BytesIO 

st.set_page_config(page_title = 'Dashboard Planes de Salud ', layout="wide")

st.title("Dashboard interactivo: Análisis Planes de Salud")

## Carga del dataset

@st.cache_data
def cargar_datos(archivo):
    return pd.read_csv(archivo)


df = cargar_datos("df_subset.csv")


st.write("Vista previa del dataframe")
st.dataframe(df.head())
st.write(df.shape)


###Filtros interactivos

st.sidebar.header("Diagnóstico de Filtros")

with st.sidebar.expander("Ver valores disponibles"):
    st.write("Modalidad:", df['ModalidadAtencion'].unique())
    st.write("Tipos de Plan:", df['TipoPlan'].unique())
    st.write("Rango ValorPlan:", df['ValorPlan'].min(), "-", df['ValorPlan'].max())
    st.write("Comercialización:", df['Comercializacion'].unique())

st.sidebar.header('Filtros del Panel')

# 1. Modalidad con opción "Todas" 
modalidad_opciones = ['TODAS LAS MODALIDADES'] + sorted(df['ModalidadAtencion'].dropna().unique().tolist())
modalidad = st.sidebar.selectbox(
    'Modalidad de atención',
    options=modalidad_opciones,
    index=0
)

# 2. Tipo de Plan 
tipo_plan_opciones = sorted(df['TipoPlan'].dropna().unique().tolist())
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button('Seleccionar Todos'):
        tipo_plan = tipo_plan_opciones
with col2:
    if st.button('Deseleccionar Todos'):
        tipo_plan = []

tipo_plan = st.sidebar.multiselect(
    'Tipo de plan',
    options=tipo_plan_opciones,
    default=tipo_plan_opciones
)

# 3. Valor del plan en UF 
valor_uf = 39265   #valor 14-07
df['ValorPlan_UF'] = df['ValorPlan'] / valor_uf  

valor_plan_uf = st.sidebar.slider(
    'Rango de valor del plan (UF)',
    min_value=0.0,
    max_value=10.0,
    value=(0.0, min(10.0, float(df['ValorPlan_UF'].max()))),  # Máximo 10 UF
    step=0.1,
    format="%.1f"
)

#equivalencia en CLP
st.sidebar.caption(f"Equivalente en CLP: {valor_plan_uf[0]*valor_uf:,.0f} - {valor_plan_uf[1]*valor_uf:,.0f} $")

# 4. Comercialización 
comercializacion_opciones = ['TODAS LAS OPCIONES'] + sorted(df['Comercializacion'].dropna().unique().tolist())
comercializacion = st.sidebar.selectbox(
    'Estado de comercialización',
    options=comercializacion_opciones,
    index=0
)

# Botón para resetear filtros 
if st.sidebar.button('Resetear Filtros'):
    st.rerun()

# Aplicar filtros de forma condicional
df_filtrado = df.copy()

# Filtro por Modalidad 
if modalidad != 'TODAS LAS MODALIDADES':
    df_filtrado = df_filtrado[df_filtrado['ModalidadAtencion'] == modalidad]

# Filtro por Tipo de Plan 
if tipo_plan:
    df_filtrado = df_filtrado[df_filtrado['TipoPlan'].isin(tipo_plan)]

# Filtro por Rango de Valor
df_filtrado = df_filtrado[
    df_filtrado['ValorPlan_UF'].between(valor_plan_uf[0], valor_plan_uf[1])
]

# Filtro por Comercialización 
if comercializacion != 'TODAS LAS OPCIONES':
    df_filtrado = df_filtrado[df_filtrado['Comercializacion'] == comercializacion]

# Verificación final del filtrado 
if df_filtrado.empty:
    st.warning("""
    No se encontraron registros con los filtros actuales. 
    **Sugerencias:**
    1. Amplía el rango de valores
    2. Selecciona más tipos de plan
    3. Revisa las combinaciones de filtros
    """)
    df_filtrado = df.copy()
else:
    st.success(f"{len(df_filtrado)} registros encontrados")


###Crear menu

menu = st.selectbox("Selecciona una sección", ["Análisis General", "Exploración con PyGWalker"]) 

if menu == "Análisis General":
    # Acá crearemos la sección general
    
    st.write("Análsis General")
   

 ### KPIs

 #Total de Planes, Valor Promedio del Plan, Promedio % Cotización

with st.expander("Utilidad de los KPIs:"):
       st.markdown(
    """
 - Total de Planes: Te dice cuántas opciones de planes tienes disponibles o vigentes en el filtro aplicado. Sin esta cifra, los demás indicadores pierden contexto, porque el número de planes afecta todo análisis.

 - Valor Promedio del Plan: Complementa el total, mostrando cuánto cuesta en promedio cada plan. Esto ayuda a entender si la oferta es más cara o accesible, y si hay cambios relevantes en precios cuando aplicas distintos filtros (por ejemplo, por modalidad o región).

 - Promedio % Cotización: Muestra la carga relativa que representa el valor de esos planes en relación con la base de cotización (o ingreso) de los usuarios. Es fundamental para medir qué tan "pesados" son esos planes para el bolsillo del cotizante.

 En conjunto sirven para:

 - Evaluar de manera más rápida la cantidad y costo promedio de planes disponibles.

 - Medir el impacto financiero que esos planes representan para quienes pagan la cotización.

 - Detectar posibles desequilibrios o problemas, como si hay pocos planes pero muy caros, o si la carga porcentual es demasiado alta.
    """)


columna1, columna2, columna3 = st.columns(3)

total_planes = len(df_filtrado)
valor_promedio = round(df_filtrado['ValorCLP'].mean(), 0) if total_planes > 0 else 0
prom_cotizacion = round(df_filtrado['%Cotizacion'].mean(), 2) if total_planes > 0 else 0

columna1.metric("Total de Planes", total_planes)
columna2.metric("Valor Promedio del Plan", f"${valor_promedio:,.0f}")
columna3.metric("Promedio % Cotización", f"{prom_cotizacion}%")

 #Planes Restringidos, Valor Promedio Plan Restringido, % Planes Restringidos

with st.expander ("Utilidad de los KPIs:"):
     st.markdown("""
 Este análisis nos muestra cuántos planes tienen limitaciones en sus coberturas y cuánto cuestan en promedio. 
 Si esos planes con restricciones no son mucho más económicos que los completos, 
 significa que los usuarios podrían estar pagando casi lo mismo pero con menos beneficios. 
 Además, al ver qué porcentaje representan dentro de toda la oferta, 
 entendemos qué tan variada y adaptada está la cartera a diferentes necesidades.
 Esta información es muy útil para ajustar los precios y coberturas,
 asegurando que los clientes obtengan un equilibrio justo entre lo que pagan y los servicios que reciben.
 """)

columna1, columna2, columna3 = st.columns(3)

planes_restringidos = df_filtrado[df_filtrado['PrestacionesRestringidas'] == 1]['CodigoPlan'].nunique()
valor_prom_plan_restringido = round(df_filtrado[df_filtrado['PrestacionesRestringidas'] == 1]['ValorCLP'].mean(), 0) if planes_restringidos > 0 else 0
pct_planes_restringidos = round((planes_restringidos / total_planes) * 100, 2) if total_planes > 0 else 0

columna1.metric("Planes Restringidos", planes_restringidos)
columna2.metric("Valor Promedio Plan Restringido", f"${valor_prom_plan_restringido:,.0f}")
columna3.metric("% Planes Restringidos", f"{pct_planes_restringidos}%")


 ###Gráficos

 ##Distribución de planes por Estado de Comercialización

# Transformación de datos 
df_plot = df_filtrado.copy()
df_plot['Comercializacion'] = df_plot['Comercializacion'].replace({
'SI COMER.': 'Comercializado',
'NO COMER.': 'No Comercializado'
 }).fillna('Sin Datos')

 # Crear el gráfico SOLO si hay datos
if not df_plot.empty:
    st.subheader("Distribución por Estado de Comercialización")
    
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(
        data=df_plot,
        x='Comercializacion',
        order=df_plot['Comercializacion'].value_counts().index,
        palette='viridis',
        saturation=0.8
    )
    
    # Personalización del gráfico
    plt.title('Distribución de Planes por Estado de Comercialización', pad=20, fontweight='bold')
    plt.xlabel('Estado de Comercialización')
    plt.ylabel('Cantidad de Planes')
    
    # Añadir etiquetas de valor
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}', 
            (p.get_x() + p.get_width()/2., p.get_height()),
            ha='center', va='center', 
            xytext=(0, 5), 
            textcoords='offset points',
            fontweight='bold'
        )
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(plt)
    
    # Mostrar tabla resumen
    with st.expander("📊 Ver datos detallados"):
        st.dataframe(
            df_plot['Comercializacion'].value_counts().reset_index().rename(
                columns={'index': 'Estado', 'Comercializacion': 'Cantidad'}
            ),
            hide_index=True
        )
else:
    st.error("No hay datos disponibles para mostrar el gráfico")           


st.info("""
 El gráfico anterior muestra la distribución de planes de salud según su estado de comercialización, 
 mostrando una diferencia significativa entre los planes que se encuentran disponibles en comercialización y los que no lo están. 
 Se logra visualizar que de un total de 2.000 planes de salud, solo 198 de encuentran en comercialización, mientras que el resto, 
 es decir, 1.802 planes no se encuentran disponibles y no de comercializan para el público. 
 Esta información puede servir para evaluar que tan eficiente es la oferta comercial o también para revisar que planes  pueden activarse o 
 actualizarse para que se comercialicen.
 """)



 ## Distribución por Modalidad de Atención

st.subheader("Distribución por Modalidad de Atención")

df_plot = df_filtrado if not df_filtrado.empty else df

 #Procesamiento de datos
modalidad_counts = df_plot['ModalidadAtencion'].value_counts().reset_index()
modalidad_counts.columns = ['Modalidad', 'Cantidad']

 #Ordenar por cantidad descendente
modalidad_counts = modalidad_counts.sort_values('Cantidad', ascending=False)

 #Creación del gráfico
plt.figure(figsize=(10, 6))  # Tamaño aumentado para mejor visualización
ax = sns.barplot(
    data=modalidad_counts,
    x='Modalidad',
    y='Cantidad',
    palette='Set2',
    saturation=0.8
 )

 #Personalización del gráfico
plt.title('Distribución por Modalidad de Atención', 
          fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Modalidad de Atención', fontsize=12, labelpad=10)
plt.ylabel('Cantidad de Planes', fontsize=12, labelpad=10)
plt.xticks(rotation=15)

 #Añadir etiquetas de valor 
for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height())}', 
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='center', 
        xytext=(0, 5), 
        textcoords='offset points',
        fontsize=11,
        fontweight='bold',
        color='black'
    )

plt.tight_layout()
st.pyplot(plt)

 #Mostrar tabla de datos 
with st.expander("Ver datos detallados"):
    st.dataframe(
        modalidad_counts,
        column_config={
            "Modalidad": "Modalidad de Atención",
            "Cantidad": st.column_config.NumberColumn("Cantidad", format="%d")
        },
        hide_index=True
    )

 # 8. Mensaje informativo si se usaron datos sin filtrar
if df_filtrado.empty:
    st.info("Se muestran todos los datos disponibles (no se aplicaron filtros)")


st.info("""
  El gráfico muestra cómo se distribuyen los planes de salud según la modalidad de atención en un sistema de salud o prestación de servicios.
  Se pueden identificar tres modalidades: “Prestador Preferente”, “Libre Elección” y “Plan Cerrado”.  
  Al observar el grafico se puede notar que “Prestador Preferente” es la más común, el cual representa la mayor cantidad con 998 planes.
  Luego le sigue “Libre Elección” con 884 planes, y por ultimo, se observa que “Plan Cerrado” presenta la menor participación con 118 planes.
  Esta visualización permite comprender, de forma clara, qué tipo de atención es más habitual entre los planes disponibles, 
  lo cual puede ser útil para evaluar las preferencias del sistema y orientar futuras decisiones estratégicas. 
  Se observa que la gran mayoría de planes se concentran en modalidades que ofrecen mayor flexibilidad al usuario, 
  lo que refleja una tendencia hacia esquemas menos restrictivos. Por otro lado, 
  se añade un filtro interactivo que permite explorar los datos de manera dinámica, 
  facilitando el análisis y comparación entre las diferentes modalidades de atención.
 """)


 ##Distribución de Tipos de Plan

st.subheader("Distribución de Tipos de Plan")

df_plot = df_filtrado if not df_filtrado.empty else df
tipo_counts = df_plot['TipoPlan'].value_counts()

 #Creación del gráfico 
fig, ax = plt.subplots(figsize=(8, 8))  # Tamaño aumentado para mejor visualización

 # Paleta de colores 
colors = sns.color_palette("pastel", n_colors=len(tipo_counts))

wedges, texts, autotexts = ax.pie(
    tipo_counts,
    labels=tipo_counts.index,
    autopct=lambda p: f'{p:.1f}%\n({int(round(p*sum(tipo_counts)/100))})',
    startangle=140,
    colors=colors,
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
    textprops=dict(color="black", fontsize=12, fontweight='bold'),
    pctdistance=0.85  # Mueve los porcentajes hacia el centro
 )

 # Mejoras visuales
plt.setp(autotexts, size=12, weight="bold")
plt.setp(texts, size=12, weight="bold")

 # Añadir título y leyenda
ax.set_title('Distribución de Tipo de Plan\n(Grupal vs Individual)', 
             fontsize=16, pad=20, weight='bold')

 # Crear leyenda fuera del gráfico
ax.legend(
    wedges,
    tipo_counts.index,
    title="Tipos de Plan",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=12
 )

plt.tight_layout()
st.pyplot(fig)

 #Mostrar tabla de datos
with st.expander("Ver datos detallados"):
    st.dataframe(
        tipo_counts.reset_index().rename(columns={'index': 'Tipo de Plan', 'TipoPlan': 'Cantidad'}),
        hide_index=True,
        use_container_width=True
    )

 #Mensaje informativo si se usaron datos sin filtrar
if df_filtrado.empty:
    st.info("Se muestran todos los datos disponibles (no se aplicaron filtros)")


st.info("""
  El gráfico nos muestra la distribución de los planes de salud según el tipo de plan, diferenciando entre “Individual” y “Grupal”. 
  Observamos que los planes individuales son mayoritarios, con el total de 1.267 planes, lo que representa el 63,3% del total. 
  Por lo contrario los planes grupales suman 733, que es del 36,6%. 
  Esta visualización nos permite identificar qué tipo de plan es más habitual dentro del sistema.
 """)


 ##Valor Promedio del Plan por Modalidad

st.subheader("Valor Promedio del Plan por Modalidad")

df_plot = df_filtrado if not df_filtrado.empty else df
valor_por_modalidad = df_plot.groupby('ModalidadAtencion')['ValorPlan'].mean().sort_values(ascending=False)

 #Creación del gráfico
fig, ax = plt.subplots(figsize=(10, 6))  

barplot = sns.barplot(
    x=valor_por_modalidad.index,
    y=valor_por_modalidad.values,
    palette='plasma',
    ax=ax,
    saturation=0.85
 )

 #Personalización del gráfico
ax.set_title('Valor Promedio del Plan por Modalidad de Atención', 
             fontsize=16, pad=20, fontweight='bold')
ax.set_xlabel('Modalidad de Atención', fontsize=12, labelpad=10)
ax.set_ylabel('Valor Promedio (UF)', fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha='right')

 #Añadir etiquetas de valor
for p in ax.patches:
    ax.annotate(
        f'{p.get_height():.1f} UF',  # Mostrar con 1 decimal
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='center', 
        xytext=(0, 5), 
        textcoords='offset points',
        fontsize=11,
        fontweight='bold'
    )

plt.tight_layout()
st.pyplot(fig)

 #Mostrar tabla de datos
with st.expander("Ver datos detallados"):
    st.dataframe(
        valor_por_modalidad.reset_index().rename(columns={
            'ModalidadAtencion': 'Modalidad',
            'ValorPlan': 'Valor Promedio (UF)'
        }),
        hide_index=True,
        column_config={
            "Valor Promedio (UF)": st.column_config.NumberColumn(
                format="%.1f UF"
            )
        }
    )

 #Mensaje informativo si se usaron datos sin filtrar
if df_filtrado.empty:
    st.info("Se muestran todos los datos disponibles (no se aplicaron filtros)")   



st.info("""
 Este gráfico compara el valor promedio de los planes de salud de acuerdo a tres modalidades de atención,
  las cuales son: “Libre Elección”, “Prestador Preferente” y “Plan Cerrado”. 
  Se observa una diferencia significativa en el costo promedio entre ellas. 
  “Libre Elección” es la modalidad con el valor más alto, alcanzando 3,9 UF, 
  lo que sugiere mayor libertad para elegir prestadores, pero a un precio más elevado. 
  “Prestador Preferente” presenta un valor medio de 1,8 UF, mientras que “Plan Cerrado” es el más económico, 
  con solo 1,2 UF, aunque probablemente implique restricciones importantes en cuanto a prestadores disponibles. 
  Esta comparación permite visualizar cómo el grado de libertad en la atención médica impacta directamente en el costo del plan.
 """)



 ##Distribución del Valor del Plan por Modalidad

st.subheader("Distribución del Valor del Plan por Modalidad")
df_plot = df_filtrado if not df_filtrado.empty else df

 #Configuración del gráfico 
plt.figure(figsize=(12, 7))
ax = sns.boxplot(
    x='ModalidadAtencion',
    y='ValorPlan',
    data=df_plot,
    palette='viridis',
    showmeans=False  
 )

 #Personalización del gráfico
plt.title('Distribución del Valor del Plan (UF) por Modalidad', 
          fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Modalidad de Atención', fontsize=12, labelpad=10)
plt.ylabel('Valor del Plan (UF)', fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha='right')

 #Establecer límites del eje Y
y_upper_limit = df_plot['ValorPlan'].quantile(0.95)
plt.ylim(0, y_upper_limit)
st.pyplot(plt.gcf())

 #Explicación del gráfico
st.info("""
 Características del gráfico:
 - La caja representa el rango intercuartílico (50% central de los datos)
 - La línea dentro de la caja es la mediana
 - Los bigotes muestran el rango de valores típicos
 """)

st.info("""
 El gráfico muestra la distribución del valor de los planes de salud (UF) según su modalidad.
 Observamos que los planes de Libre Elección tienen los valores más altos y dispersos, 
 mientras que los "Preferentes" presentan valores intermedios con algunos casos extremos.
 Por último los planes de Cerrado concentran los valores más bajos y menos variables. 
 Esta información nos muestra las diferencias de precios según la modalidad y analizar posibles ajustes en la oferta
 """)




 ##Proporción de planes cuya cotización supera el 7% del ingreso estimado

 # Configuración 
st.set_page_config(
    page_title="Análisis de Planes en UF",
    layout="wide",  # Cambiado a wide
    initial_sidebar_state="expanded"
 )

 # CSS personalizado
st.markdown("""
 <style>
    .main .block-container {
        max-width: 90vw;
    }
    .stPlot {
        width: 100% !important;
    }
 </style>
 """, unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuración")
    ingreso_uf = st.number_input("Ingreso mensual de referencia (UF)", 
                               min_value=1.0, value=27.03, step=0.5)
    umbral = st.slider("Umbral porcentual crítico", 1, 15, 7)

 #Cálculos
df['%Cotizacion'] = (df['ValorPlan'] / ingreso_uf) * 100
df[f'Sobre{umbral}'] = df['%Cotizacion'] > umbral

 #Métricas en columnas ajustadas
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Total Planes", len(df))
with m2:
    st.metric("Valor Promedio", f"{df['ValorPlan'].mean():.2f} UF")
with m3:
    st.metric(f"> {umbral}% del ingreso", 
             f"{df[f'Sobre{umbral}'].sum()} ({df[f'Sobre{umbral}'].mean()*100:.1f}%)")

 # Gráfico Plotly 
st.plotly_chart(px.pie(
    df, 
    names=df[f'Sobre{umbral}'].map({True: f'> {umbral}%', False: f'≤ {umbral}%'}),
    title=f"Distribución respecto al {umbral}% del ingreso",
    hole=0.4,
    width=800,
    height=500
 ), use_container_width=True)


st.info("""
Para el análisis del porcentaje de cotización que representa el valor de los planes de salud, 
se ha utilizado un ingreso referencial de $1.000.000. Este valor se aproxima al ingreso mediano mensual en Chile, 
lo que permite estimar qué proporción del sueldo se destinaría al pago del plan.
""")


st.info("""
 Este gráfico muestra la distribución de 2000 planes en relación con el 7% del ingreso.
 Se logra observar que el valor promedio de los planes es de un 2,75 UF, 
 Lo más llamativo es que 988 de estos planes, lo que equivale a un 49,4% del total,
 están igual o por debajo del umbral del cual es el 7% del ingreso, Los restantes 1012 planes (50,6%) superan ese porcentaje.
 Esta grafico circular, permite entender de una manera rápida y exacta cuántos planes cumplen con el criterio según el 7% del ingreso, 
 lo cual resulta relevante para evaluar su accesibilidad financiera.
 """)




 ## Distribución de planes por cobertura geográfica
try:
    st.write("Distribución de planes por cobertura geográfica")
    
 #Verificar si la columna existe 
    col_name = next((col for col in df.columns if 'regioncomerciali' in col.lower()), None)
    
    if col_name is None:
        st.error("Error: No se encontró la columna de regiones")
        st.write("Columnas disponibles:", df.columns.tolist())
    else:
 #Procesamiento seguro
        df_regiones = df.copy()
        df_regiones['Region'] = df_regiones[col_name].astype(str)
        
 #Mostrar diagnóstico
        with st.expander("Ver valores únicos antes de procesar"):
            st.write(df_regiones['Region'].unique())
        
 #Separar regiones si están concatenadas
        if df_regiones['Region'].str.contains(',').any():
            df_regiones['Region'] = df_regiones['Region'].str.split(',')
            df_regiones = df_regiones.explode('Region')
        
        df_regiones['Region'] = df_regiones['Region'].str.strip()
        
 #Conteo y ordenamiento
        region_counts = df_regiones['Region'].value_counts().reset_index()
        region_counts.columns = ['Región', 'Cantidad']
        region_counts = region_counts.sort_values('Cantidad', ascending=False)
        
 #Crear gráfico
        if not region_counts.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=region_counts,
                y='Región',
                x='Cantidad',
                palette='viridis'
            )
            
 #Personalización
            plt.title('Distribución de Planes por Región', fontsize=14)
            plt.xlabel('Cantidad de Planes', fontsize=12)
            plt.ylabel('')
            
 #Añadir etiquetas
            for p in ax.patches:
                ax.text(
                    p.get_width() + 0.5,
                    p.get_y() + p.get_height()/2,
                    f'{int(p.get_width())}',
                    va='center'
                )
            
            st.pyplot(fig)
            
            
 #Mostrar tabla
            with st.expander("📊 Ver datos completos"):
                st.dataframe(region_counts)
        else:
            st.warning("No se encontraron datos para graficar")

except Exception as e:
    st.error(f"Error inesperado: {str(e)}")



st.info("""
 Este es un gráfico de barras horizontales muestra la cantidad de planes disponibles según su cobertura geográfica. 
 Se puede notar que mayor parte corresponde a planes de cobertura nacional, con un total de 1.278 planes, 
 luego con se encuentra los que tienen una cobertura con mezcla de regiones con 503 planes, 
 seguidos por los planes existentes solo en la Región Metropolitana (RM) con 198 planes, y finalmente, 
 los planes sin comercialización que son 21. 
 Este gráfico permite visualizar de forma clara que los planes con cobertura nacional predominan sobre otras categorías.
 Además, cuenta con un filtro interactivo que permite desglosar y observar la distribución de los planes específicos por cada área geográfica, 
 brindando una herramienta útil para el análisis detallado y personalizado según el área de interés.
 """)


 ##Evolución Histórica de Planes

 #Configuración de página 
st.set_page_config(page_title="Análisis Temporal", layout="wide")
st.title("Evolución Histórica de Planes")

 #Verificación de datos 
if 'FechaInicioPlan' not in df.columns:
    st.error("El dataset no contiene la columna 'FechaInicioPlan' requerida")
    st.stop()

 #Sidebar con controles
with st.sidebar:
    st.header("Filtros")
    
 #Convertir a datetime y obtener rango
    df['FechaInicioPlan'] = pd.to_datetime(df['FechaInicioPlan'], errors='coerce')
    min_date = df['FechaInicioPlan'].min().to_pydatetime()
    max_date = df['FechaInicioPlan'].max().to_pydatetime()
    
    fecha_rango = st.date_input(
        "Período de análisis",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
 #Widget para agrupamiento temporal
    agrupamiento = st.radio(
        "Agrupar por",
        options=['Mensual', 'Trimestral', 'Anual'],
        index=0
    )

 #Procesamiento de datos
freq_map = {'Mensual': 'M', 'Trimestral': 'Q', 'Anual': 'Y'}
periodo = freq_map[agrupamiento]

df_filtrado = df[
    (df['FechaInicioPlan'].dt.date >= fecha_rango[0]) & 
    (df['FechaInicioPlan'].dt.date <= fecha_rango[1])
 ]

 #Agrupamiento temporal
df_filtrado['periodo'] = df_filtrado['FechaInicioPlan'].dt.to_period(periodo)
conteo_planes = df_filtrado.groupby('periodo').size().reset_index(name='count')
conteo_planes['periodo'] = conteo_planes['periodo'].dt.to_timestamp()

 #Visualización interactiva
fig = px.area(
    conteo_planes,
    x='periodo',
    y='count',
    title=f"Planes por {agrupamiento.lower()} ({fecha_rango[0].strftime('%Y-%m-%d')} a {fecha_rango[1].strftime('%Y-%m-%d')})",
    labels={'periodo': 'Fecha', 'count': 'Número de Planes'},
    line_shape='spline',
    template='plotly_white'
 )

fig.update_layout(
    hovermode="x unified",
    height=500,
    xaxis=dict(
        rangeslider=dict(visible=True),
        tickformat=f'%b %Y' if agrupamiento != 'Anual' else '%Y'
    )
 )

 #Mostrar resultados
col1, col2 = st.columns([4, 1])
with col1:
    st.plotly_chart(fig, use_container_width=True)
    
with col2:
    st.metric("Total Planes", conteo_planes['count'].sum())
    st.metric(f"Mayor {agrupamiento}", 
             f"{conteo_planes.loc[conteo_planes['count'].idxmax(), 'count']} planes")
    st.metric("Promedio", round(conteo_planes['count'].mean(), 1))

 #Análisis complementario
with st.expander(" Detalles estadísticos"):
    st.write(conteo_planes.describe().rename(
        index={'count': 'Conteo', '50%': 'Mediana'}))

st.caption(f"Dataset cargado: {len(df)} registros totales | {len(df_filtrado)} después de filtrar")



st.info("""
 El gráfico muestra la evolución histórica de la cantidad de planes desde el año 2000 hasta 2020,
 presentando un crecimiento significativo en el año 2000, despúes se observa una decadencia, hasta que por el año 2007  
 aumenta la cantidad de planes hasta 210 aproximadamente. 
 En el año 2012 es donde hay una menor cantidad de planes, pero en el 2016 vuelve a aumentar hasta 130 aprox.
 Esto sugiere una variabilidad notable en la oferta, posiblemente vinculada a factores económicos, 
 regulatorios o de mercado.
 """)


 ###Sección PyGWalker en proceso............






### Tabla y estadísticas

tab1, tab2, = st.tabs(["Tabla", "Estadísticas"])

with tab1:
    # escribir codigo
    st.dataframe(df_filtrado)

with tab2:
    st.write('Estadisticas descriptivas')
    st.dataframe(df_filtrado.describe())


###Formulario feedback

with st.form("Formulario de Feedback"):
    st.subheader("Feedback del usuario")

    nombre = st.text_input("Tu nombre")
    comentario = st.text_area("¿Qué te pareció el dashboard?")
    puntaje = st.slider("Puntaje de satisfacción:", 1,10,5)

    enviar = st.form_submit_button("Enviar")

    if enviar:
        # enviar la información a una base de datos  
        st.success(f'Gracias {nombre}! Calificaste a nuestro dashboard con un {puntaje}/10')


# Elementos en la barra lateral
st.sidebar.markdown("---")
st.sidebar.markdown("Creado por: **Bárbara Ibarra, Gerard García, Nicolás Espinoza y Yasna Díaz**")
st.sidebar.markdown("**Ingenieros en Información y Control de Gestión**")
