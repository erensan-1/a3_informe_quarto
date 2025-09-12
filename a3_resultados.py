import pandas as pd
import geopandas as gpd

# Cargar la tabla
df = pd.read_csv("a3_asignaciones_vc_rf.csv")

# Convertir a GeoDataFrame
gdf = gpd.GeoDataFrame(
    df, 
    geometry=gpd.points_from_xy(df['coor_x'], df['coor_y']), 
    crs="EPSG:25830"
)

# Crear columna 'asignado' según 'metodo'
df['asignado'] = df['metodo'].notna() & (df['metodo'] != '')

# ---------- 1. Estadísticas generales de asignaciones ----------
print("Número total de registros:", len(df))
print("Número de registros asignados:", df['asignado'].sum())
print("Número de registros pendientes:", (~df['asignado']).sum())

# ---------- 2. Municipios con más y menos asignaciones ----------
# Conteo de registros asignados por municipio
asignaciones_municipio = df.groupby('cmun_ine')['asignado'].sum()

# Total de registros por municipio
total_municipio = df.groupby('cmun_ine').size()

# Calcular porcentaje de asignados
porcentaje_asignado = (asignaciones_municipio / total_municipio * 100).round(2)

# Crear DataFrame combinado para exportar
resumen_municipio = pd.DataFrame({
    'Registros asignados': asignaciones_municipio,
    'Total registros': total_municipio,
    'Porcentaje asignado': porcentaje_asignado
}).sort_values('Registros asignados', ascending=False)

# ---------- 2b. Pendientes por municipio ----------
pendientes_municipio = total_municipio - asignaciones_municipio
porcentaje_pendiente = (pendientes_municipio / total_municipio * 100).round(2)
resumen_pendientes_municipio = pd.DataFrame({
    'Registros pendientes': pendientes_municipio,
    'Total registros': total_municipio,
    'Porcentaje pendiente': porcentaje_pendiente
}).sort_values('Registros pendientes', ascending=False)

# ---------- 3. Direcciones con mayor número de puntos sin asignar ----------
pendientes_direccion = df[~df['asignado']].groupby('direccion_vv').size().sort_values(ascending=False)
print("Direcciones con más puntos pendientes:\n", pendientes_direccion.head(10))

# ---------- 4. Estadísticas de scores ----------
score_cols = [col for col in df.columns if col.startswith('score_')]

# ---------- 5. Conteo y porcentaje de asignaciones por método incluyendo vacíos ----------
total_registros = len(df)

# Contar cada método específico
conteo_fuzzy = (df['metodo'] == 'fuzzy + embedding + coseno').sum()
conteo_exacto = (df['metodo'] == 'match exacto').sum()
conteo_vacios = df['metodo'].isna().sum() + (df['metodo'] == '').sum()

# Crear DataFrame resumen
metodo_stats = pd.DataFrame({
    'Metodo': ['fuzzy + embedding + coseno', 'match exacto', 'No asignado'],
    'Conteo': [conteo_fuzzy, conteo_exacto, conteo_vacios],
})

# Calcular porcentaje
metodo_stats['Porcentaje'] = (metodo_stats['Conteo'] / total_registros * 100).round(2)

# ---------- 6. Exportar resultados a Excel ----------
with pd.ExcelWriter("a3_resultados.xlsx") as writer:
    resumen_municipio.to_excel(writer, sheet_name="Asignaciones por municipio")
    resumen_pendientes_municipio.to_excel(writer, sheet_name="Pendientes por municipio")
    pendientes_direccion.to_excel(writer, sheet_name="Pendientes por direccion")
    df[score_cols].describe().to_excel(writer, sheet_name="Stats scores")
    metodo_stats.to_excel(writer, sheet_name="Asignaciones por metodo")

print("Análisis completo y exportado a 'a3_resultados.xlsx'.")
