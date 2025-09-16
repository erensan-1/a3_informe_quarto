import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 1. Conexión ----------
DB_URI = "postgresql+psycopg2://erensan:5eP79ECt@192.168.21.226:5432/padron_online"
engine = create_engine(DB_URI)
esquema = 'ine'

# ---------- 2. Leer tablas desde la BBDD ----------
vv = pd.read_sql(f"SELECT cvia_dgc, cmun_dgc, cvia_ine, tvia_dgc, nvia_dgc FROM {esquema}.padrononline_vv", engine)

# Filtrar viviendas sin CVIA pero con datos en tvia_dgc o nvia_dgc
vv_problem = vv[(vv['cvia_ine'] == 0) & ((vv['tvia_dgc'].notna()) | (vv['nvia_dgc'].notna()))].copy()
vv_problem['cmun_dgc'] = vv_problem['cmun_dgc'].astype(str).str.zfill(3)

# ---------- 3. Diccionario municipios ----------
municipios_tenerife = {
    '001': 'Adeje', '004': 'Arafo', '005': 'Arico', '006': 'Arona',
    '010': 'Buenavista del Norte', '011': 'Candelaria', '012': 'Fasnia',
    '015': 'Garachico', '017': 'Granadilla de Abona', '018': 'Guancha, La',
    '019': 'Guía de Isora', '020': 'Güímar', '022': 'Icod de los Vinos',
    '025': 'Matanza de Acentejo, La', '026': 'Orotava, La', '028': 'Puerto de la Cruz',
    '031': 'Realejos, Los', '032': 'Rosario, El', '023': 'San Cristóbal de La Laguna',
    '034': 'San Juan de la Rambla', '035': 'San Miguel de Abona', '038': 'Santa Cruz de Tenerife',
    '039': 'Santa Úrsula', '040': 'Santiago del Teide', '041': 'Sauzal, El',
    '042': 'Silos, Los', '043': 'Tacoronte', '044': 'Tanque, El',
    '046': 'Tegueste', '051': 'Victoria de Acentejo, La', '052': 'Vilaflor de Chasna'
}

vv_problem['municipio'] = vv_problem['cmun_dgc'].map(municipios_tenerife)

# ---------- 4. Conteos básicos ----------
total_problem = len(vv_problem)
print(f"Registros problemáticos: {total_problem}")

# ---------- 5. Distribución por municipio ----------
vv_municipio = (
    vv_problem
    .groupby('cmun_dgc')
    .agg(
        total=('cmun_dgc', 'size'),
        vias_distintas=('tvia_dgc', 
                        lambda x: pd.Series(x).str.cat(
                            vv_problem.loc[x.index, 'nvia_dgc'].astype(str),
                            sep='', na_rep=''
                        ).nunique()
                       )
    )
    .reset_index()
)

# Añadir nombre del municipio
vv_municipio['Municipio'] = vv_municipio['cmun_dgc'].map(municipios_tenerife)

# Renombrar y reordenar columnas
vv_municipio = vv_municipio.rename(
    columns={
        'cmun_dgc': 'Código',
        'total': 'Nº viviendas',
        'vias_distintas': 'Nº vías'
    }
)[['Código', 'Municipio', 'Nº viviendas', 'Nº vías']]

# Crear fila de totales
totales = pd.DataFrame([{
    'Código': 'TOTAL',
    'Municipio': '',
    'Nº viviendas': vv_municipio['Nº viviendas'].sum(),
    'Nº vías': vv_municipio['Nº vías'].sum()
}])

# Concatenar tabla con la fila de totales
vv_municipio = pd.concat([vv_municipio, totales], ignore_index=True)


# ---------- 6. Distribución por tipo de vía ----------
vv_tipo_via = vv_problem.groupby('tvia_dgc').agg(total=('tvia_dgc', 'size')).reset_index().sort_values('total', ascending=False)

# ---------- 7. Direcciones duplicadas ----------
vv_problem['direccion'] = vv_problem['tvia_dgc'].fillna('') + " " + vv_problem['nvia_dgc'].fillna('')

# Número de registros por dirección
registros_por_direccion = vv_problem.groupby('direccion').size().reset_index(name='num_registros')
registros_por_direccion = registros_por_direccion.sort_values('num_registros', ascending=False)

# Crear hoja adicional con código de municipio concatenado a la dirección
registros_por_direccion_mun = vv_problem.copy()
registros_por_direccion_mun['direccion_mun'] = vv_problem['cmun_dgc'] + " " + vv_problem['direccion']
registros_por_direccion_mun = registros_por_direccion_mun.groupby('direccion_mun').size().reset_index(name='num_registros')
registros_por_direccion_mun['municipio'] = registros_por_direccion_mun['direccion_mun'].str[:3].map(municipios_tenerife)
registros_por_direccion_mun = registros_por_direccion_mun.sort_values('num_registros', ascending=False)

# ---------- 8. Registros incompletos ----------
incompletos_tvia = vv_problem['tvia_dgc'].isna().sum()
incompletos_nvia = vv_problem['nvia_dgc'].isna().sum()
print(f"Registros con tvia_dgc vacío: {incompletos_tvia}")
print(f"Registros con nvia_dgc vacío: {incompletos_nvia}")

# ---------- 9. Estadísticas numéricas ----------
nvia_stats = pd.to_numeric(vv_problem['nvia_dgc'], errors='coerce').dropna()
nvia_describe = nvia_stats.describe()
print("Estadísticas de nvia_dgc (solo números presentes):")
print(nvia_describe)

# ---------- 10. Registros por vía ----------
# Global
vv_problem['direccion_completa'] = vv_problem['cmun_dgc'] + " " + vv_problem['tvia_dgc'].fillna('') + " " + vv_problem['nvia_dgc'].fillna('')
registros_por_via_global = vv_problem.groupby('cvia_dgc').size().reset_index(name='num_registros')
registros_por_via_global['direccion'] = vv_problem.groupby('cvia_dgc')['direccion_completa'].first().values
registros_por_via_global = registros_por_via_global.sort_values('num_registros', ascending=False)

# Por municipio
registros_por_via_municipio = vv_problem.groupby(['cmun_dgc','cvia_dgc']).size().reset_index(name='num_registros')
registros_por_via_municipio['direccion'] = registros_por_via_municipio.apply(
    lambda x: vv_problem.loc[
        (vv_problem['cmun_dgc']==x['cmun_dgc']) & (vv_problem['cvia_dgc']==x['cvia_dgc']), 'direccion_completa'
    ].iloc[0], axis=1)
registros_por_via_municipio['municipio'] = registros_por_via_municipio['cmun_dgc'].map(municipios_tenerife)
registros_por_via_municipio = registros_por_via_municipio.sort_values(['cmun_dgc','num_registros'], ascending=[True, False])

# ---------- 11. Guardar todas las hojas en Excel ----------
with pd.ExcelWriter('a3_diagnostico.xlsx', engine='openpyxl') as writer:
    vv_municipio.to_excel(writer, sheet_name='resumen_municipio', index=False)
    registros_por_direccion.to_excel(writer, sheet_name='registros_por_direccion', index=False)
    registros_por_direccion_mun.to_excel(writer, sheet_name='registros_por_direccion_mun', index=False)
    vv_tipo_via.to_excel(writer, sheet_name='resumen_tipo_via', index=False)
    registros_por_via_global.to_excel(writer, sheet_name='registros_por_via_global', index=False)
    registros_por_via_municipio.to_excel(writer, sheet_name='registros_por_via_municipio', index=False)

print("Archivo 'a3_diagnostico.xlsx' creado con todas las hojas de resumen y registros, usando espacios en las concatenaciones.")
