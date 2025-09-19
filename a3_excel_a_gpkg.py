import re
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

# Funciones de corrección
def corregir_coor_x(x):
    x_str = str(int(x)) if pd.notnull(x) else ""
    if re.match(r"^-?\d{8}$", x_str):
        return x / 100
    elif re.match(r"^-?\d{7}$", x_str):
        return x / 10
    else:
        return x

def corregir_coor_y(y):
    y_str = str(int(y)) if pd.notnull(y) else ""
    if re.search(r"\d{9}", y_str):
        return y / 100
    else:
        return y

# Cargar dataframe
df = pd.read_csv("a3_asignaciones_vc.csv")

# Aplicar correcciones
df['coor_x_corr'] = df['coor_x'].apply(corregir_coor_x)
df['coor_y_corr'] = df['coor_y'].apply(corregir_coor_y)

# Asignar CRS según coordenadas originales
def crs_por_coor_x(x):
    if x < 0:
        return 'EPSG:25830'
    else:
        return 'EPSG:32628'

df['crs'] = df['coor_x'].apply(crs_por_coor_x)

# Crear geometría con coordenadas corregidas
df['geometry'] = df.apply(lambda r: Point(r['coor_x_corr'], r['coor_y_corr']), axis=1)

# Convertir a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry='geometry')

# Reproyectar cada grupo a EPSG:32628
gdf_corrected = gpd.GeoDataFrame(columns=gdf.columns)

for crs_val, group in gdf.groupby('crs'):
    group = group.set_crs(crs_val, allow_override=True)
    group = group.to_crs('EPSG:32628')  # <-- corregido
    gdf_corrected = pd.concat([gdf_corrected, group])

gdf_corrected.reset_index(drop=True, inplace=True)
gdf_corrected = gdf_corrected.set_geometry('geometry')
gdf_corrected = gdf_corrected.set_crs('EPSG:32628', allow_override=True)

# Eliminar columna 'geom' si existe
if 'geom' in gdf_corrected.columns:
    gdf_corrected = gdf_corrected.drop(columns=['geom'])

# Convertir otras columnas problemáticas a string si es necesario
for col in gdf_corrected.select_dtypes(['object']).columns:
    if col != 'geometry':
        gdf_corrected[col] = gdf_corrected[col].astype(str)

# Guardar en GeoPackage listo para QGIS
gdf_corrected.to_file("a3_asignaciones_vc.gpkg", driver="GPKG")
