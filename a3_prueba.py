import pandas as pd
import geopandas as gpd

# Cargar la tabla
df_vc = pd.read_csv("a3_asignaciones_vc_rf.csv")

# Convertir a GeoDataFrame
gdf = gpd.GeoDataFrame(df_vc, geometry=gpd.points_from_xy(df_vc['coor_x'], df_vc['coor_y']), crs="EPSG:25830")

# === 1. Columna 'asignado' ===
df_vc['asignado'] = df_vc['a3_metodo'].notna() & (df_vc['a3_metodo'] != '')

# === 2. Estadísticas generales ===
total_registros = len(df_vc)
total_asignados = df_vc['asignado'].sum()
total_pendientes = total_registros - total_asignados

print("=== ESTADÍSTICAS GENERALES ===")
print(f"Total registros: {total_registros}")
print(f"Total asignados: {total_asignados}")
print(f"Total pendientes: {total_pendientes}")
print(f"% asignados: {total_asignados / total_registros * 100:.2f}%")
print(f"% pendientes: {total_pendientes / total_registros * 100:.2f}%\n")

# === 3. Asignaciones por municipio ===
asignaciones_mun = df_vc.groupby('cmun_ine')['asignado'].sum()
total_mun = df_vc.groupby('cmun_ine').size()
porcentaje_mun = (asignaciones_mun / total_mun * 100).round(2)

resumen_municipio = pd.DataFrame({
    'Registros asignados': asignaciones_mun,
    'Total registros': total_mun,
    'Porcentaje asignado': porcentaje_mun
}).sort_values('Registros asignados', ascending=False)

# Pendientes por municipio
pendientes_mun = total_mun - asignaciones_mun
porcentaje_pendientes_mun = (pendientes_mun / total_mun * 100).round(2)
resumen_pendientes_mun = pd.DataFrame({
    'Registros pendientes': pendientes_mun,
    'Total registros': total_mun,
    'Porcentaje pendiente': porcentaje_pendientes_mun
}).sort_values('Registros pendientes', ascending=False)

# === 4. Conteo por método ===
metodo_counts = df_vc['a3_metodo'].fillna('No asignado').value_counts()
metodo_pct = (metodo_counts / total_registros * 100).round(2)
resumen_metodo = pd.DataFrame({
    'Metodo': metodo_counts.index,
    'Conteo': metodo_counts.values,
    'Porcentaje': metodo_pct.values
})

# === 5. Conteo por estado ===
estado_counts = df_vc['a3_estado'].value_counts()
estado_pct = (estado_counts / total_registros * 100).round(2)
resumen_estado = pd.DataFrame({
    'Estado': estado_counts.index,
    'Conteo': estado_counts.values,
    'Porcentaje': estado_pct.values
})

# === 6. Direcciones con más puntos pendientes ===
pendientes_dir = df_vc[~df_vc['asignado']].groupby('direccion_norm_vc').size().sort_values(ascending=False)
print("Direcciones con más puntos pendientes:\n", pendientes_dir.head(10))

# === 7. Estadísticas de predict_proba ===
if 'predict_proba' in df_vc.columns:
    proba_stats = df_vc['predict_proba'].describe()
    print("\nEstadísticas de predict_proba:")
    print(proba_stats)

# === 8. Exportar resultados resumidos a Excel ===
with pd.ExcelWriter("a3_resumen_asignaciones.xlsx") as writer:
    resumen_municipio.to_excel(writer, sheet_name="Asignados por municipio")
    resumen_pendientes_mun.to_excel(writer, sheet_name="Pendientes por municipio")
    resumen_metodo.to_excel(writer, sheet_name="Asignaciones por metodo")
    resumen_estado.to_excel(writer, sheet_name="Asignaciones por estado")
    pendientes_dir.to_excel(writer, sheet_name="Pendientes por direccion")
    if 'predict_proba' in df_vc.columns:
        proba_stats.to_frame(name='predict_proba').to_excel(writer, sheet_name="Stats predict_proba")

print("\nResumen completo exportado a 'a3_resumen_asignaciones.xlsx'")
