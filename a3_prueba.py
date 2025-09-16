# LIBRARY
import pandas as pd
import geopandas as gpd
import random
import unicodedata
from rapidfuzz import process, fuzz
from sqlalchemy import create_engine
from tqdm import tqdm
from shapely import make_valid
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

BUFFER_DISTANCE = 100
SPECIAL_CODVIA_THRESHOLD = 90000
HEURISTIC_HIGH_THRESHOLD = 95
HEURISTIC_LOW_THRESHOLD = 60

logger.info("Conectando a la base de datos...")
DB_URI = "postgresql+psycopg2://erensan:5eP79ECt@192.168.21.226:5432/padron_online"
engine = create_engine(DB_URI)

SIGLAS_VIA = {
    'AL': 'ALAMEDA', 'AD': 'ALDEA', 'AP': 'APARTAMENTOS', 'AY': 'ARROYO', 'AV': 'AVENIDA',
    'BJ': 'BAJADA', 'BR': 'BARRANCO', 'BO': 'BARRIO', 'BL': 'BLOQUE', 'CL': 'CALLE',
    'CJ': 'CALLEJA', 'CM': 'CAMINO', 'CR': 'CARRERA', 'CS': 'CASERIO', 'CH': 'CHALET',
    'CO': 'COLONIA', 'CN': 'COSTANILLA', 'CT': 'CARRETERA', 'CU': 'CUESTA', 'ED': 'EDIFICIO',
    'EL': 'ESCALINATA', 'ES': 'ESCALERA', 'GL': 'GLORIETA', 'GR': 'GRUPO', 'LG': 'LUGAR',
    'MC': 'MERCADO', 'MN': 'MUNICIPIO', 'MZ': 'MANZANA', 'PA': 'PASEO ALTO', 'PB': 'POBLADO',
    'PD': 'PASADIZO', 'PJ': 'PASAJE', 'PL': 'PLACETA', 'PO': 'PASEO BAJO', 'PP': 'PASEO',
    'PQ': 'PARQUE', 'PR': 'PORTALES', 'PS': 'PASO', 'PT': 'PATIO', 'PU': 'PLAZUELA',
    'PZ': 'PLAZA', 'RA': 'RAMAL', 'RB': 'RAMBLA', 'RC': 'RINCONADA', 'RD': 'RONDA',
    'RP': 'RAMPA', 'RR': 'RIBERA', 'SC': 'SECTOR', 'SD': 'SENDA', 'SU': 'SUBIDA',
    'TO': 'TORRE', 'TR': 'TRAVESÍA', 'UR': 'URBANIZACIÓN', 'URB': 'URBANIZACIÓN', 'VI': 'VÍA', 'ZO': 'ZONA',
    'AVDA': 'AVENIDA', 'CTRA': 'CARRETERA', 'CLLON': 'CALLEJON', 'TRVA': 'TRAVESIA',
    'CMNO': 'CAMINO', 'PSAJE': 'PASAJE', 'BRANC': 'BARRANCO', 'PZTA': 'PLAZOLETA',
    'POLIG': 'POLIGONO', 'BARDA': 'BARRIADA', 'TRVAL': 'TRASVERSAL', 'ACCES': 'ACCESO',
    'CZDA': 'CALZADA', 'CZADA': 'CALZADA', 'VREDA': 'VEREDA', 'PG':'POLIGONO', 'FCO RGUEZ': 'FRANCISCO RODRIGUEZ'
}

municipios = {
    "38004": "Arafo", "38005": "Arico", "38010": "Buenavista del Norte", "38012": "Fasnia",
    "38015": "Garachico", "38018": "Guancha, La", "38025": "Matanza de Acentejo, La", 
    "38032": "Rosario, El", "38034": "San Juan de la Rambla", "38039": "Santa Úrsula",
    "38040": "Santiago del Teide", "38041": "Sauzal, El", "38042": "Silos, Los",
    "38044": "Tanque, El", "38046": "Tegueste", "38051": "Victoria de Acentejo, La",
    "38052": "Vilaflor de Chasna"
}

def clean_index_right(gdf):
    if 'index_right' in gdf.columns:
        return gdf.drop(columns=['index_right'])
    return gdf

def safe_sjoin(left_gdf, right_gdf, how='left', predicate='within', rsuffix=''):
    left_clean = clean_index_right(left_gdf)
    right_clean = clean_index_right(right_gdf)
    return gpd.sjoin(left_clean, right_clean, how=how, predicate=predicate, rsuffix=rsuffix)

def limpiar_texto(texto):
    if not isinstance(texto, str):
        texto = str(texto)
    texto = texto.upper().strip()
    texto = unicodedata.normalize('NFKD', texto)
    return ''.join(c for c in texto if not unicodedata.combining(c))

def normalizar_via_nombre(tvia, nvia):
    tvia_raw = SIGLAS_VIA.get(str(tvia).strip().upper(), str(tvia))
    tvia_norm = limpiar_texto(tvia_raw)
    nvia_norm = limpiar_texto(nvia)
    return f"{tvia_norm} {nvia_norm}"

def corregir_geoms(gdf, nombre):
    logger.info(f" - {nombre} inicial: {len(gdf)}")
    gdf['geom'] = gdf['geom'].apply(lambda g: make_valid(g) if g is not None else None)
    gdf = gdf[gdf.is_valid & ~gdf.is_empty]
    logger.info(f" - {nombre} válidas: {len(gdf)}")
    return gdf

def compute_features(vc, callejero):
    return {
        "ratio": fuzz.ratio(vc, callejero),
        "partial_ratio": fuzz.partial_ratio(vc, callejero),
        "token_sort_ratio": fuzz.token_sort_ratio(vc, callejero),
        "token_set_ratio": fuzz.token_set_ratio(vc, callejero),
        "wratio": fuzz.WRatio(vc, callejero),
        "len_diff": abs(len(vc) - len(callejero)),
        "first_token_equal": int(vc.split()[0] == callejero.split()[0]) if vc and callejero else 0,
        "last_token_equal": int(vc.split()[-1] == callejero.split()[-1]) if vc and callejero else 0,
    }

def filtro_heuristico(vc, callejero):
    tsr_set = fuzz.token_set_ratio(vc, callejero)
    wr = fuzz.WRatio(vc, callejero)
    if tsr_set >= HEURISTIC_HIGH_THRESHOLD:
        return 1
    elif wr < HEURISTIC_LOW_THRESHOLD:
        return 0
    else:
        return None

def decidir_match(vc, callejero, clf):
    h = filtro_heuristico(vc, callejero)
    if h is not None:
        return h
    feats = pd.DataFrame([compute_features(vc, callejero)])
    return int(clf.predict(feats)[0])

def check_poblacion_assignment(row, df_poblaciones_mun, df_vc_crs):
    poblacion = safe_sjoin(
        gpd.GeoDataFrame([row], geometry='geom', crs=df_vc_crs),
        df_poblaciones_mun[['cunn_1', 'geom']],
        how='left',
        predicate='within'
    )
    return (not poblacion.empty and pd.notna(poblacion['cunn_1'].iloc[0]))

def process_portal_matching(row, df_callejero_num_mun, df_parcela_mun, refcat):
    if pd.notna(refcat):
        parcela_geom = df_parcela_mun.loc[df_parcela_mun['refcat'] == refcat, 'geom'].iloc[0]
        cn_candidatos = df_callejero_num_mun[df_callejero_num_mun.within(parcela_geom)]
    else:
        cn_candidatos = gpd.GeoDataFrame(columns=df_callejero_num_mun.columns, geometry='geom', crs=df_callejero_num_mun.crs)
    
    if not cn_candidatos.empty:
        matches = process.extractOne(row['direccion_norm'], cn_candidatos['direccion_norm'].tolist(), scorer=fuzz.WRatio)
        if matches:
            match_name = matches[0]
            codvia = cn_candidatos.loc[cn_candidatos['direccion_norm'] == match_name, 'codvia'].iloc[0]
            uuid_match = cn_candidatos.loc[cn_candidatos['direccion_norm'] == match_name, 'uuid'].iloc[0]
            return {
                'match_name': match_name,
                'codvia': codvia,
                'uuid_match': uuid_match,
                'origen': 'callejero_num',
                'found': True
            }
    return {'found': False}

def process_via_proximity(row, df_callejero_via_mun, geom_vc):
    buffer_geom = geom_vc.buffer(BUFFER_DISTANCE)
    vias_cercanas = df_callejero_via_mun[df_callejero_via_mun.intersects(buffer_geom)]
    
    if not vias_cercanas.empty:
        matches = process.extractOne(row['direccion_norm'], vias_cercanas['direccion_norm'].tolist(), scorer=fuzz.WRatio)
        if matches:
            match_name = matches[0]
            codvia = vias_cercanas.loc[vias_cercanas['direccion_norm'] == match_name, 'codvia'].iloc[0]
            uuid_match = vias_cercanas.loc[vias_cercanas['direccion_norm'] == match_name, 'uuid'].iloc[0]
            return {
                'match_name': match_name,
                'codvia': codvia,
                'uuid_match': uuid_match,
                'origen': 'callejero_via',
                'found': True
            }
    return {'found': False}

def determine_a3_estado(row, match_result, df_poblaciones_mun, df_vc_crs, clf):
    if not match_result['found']:
        if check_poblacion_assignment(row, df_poblaciones_mun, df_vc_crs):
            return 'A3_AsignadaUnidadPoblacional'
        else:
            return 'A3_Aislada'
    
    codvia = int(match_result['codvia'])
    
    if codvia > SPECIAL_CODVIA_THRESHOLD:
        if check_poblacion_assignment(row, df_poblaciones_mun, df_vc_crs):
            return 'A3_AsignadaUnidadPoblacional'
        else:
            return 'A3_AsignarViaNueva'
    else:
        if "NONE" in str(row['direccion_norm']) or "NONE" in str(match_result['match_name']):
            return 'A3_AsignadaViaAlternativa'
        else:
            match_label = decidir_match(row['direccion_norm'], match_result['match_name'], clf)
            return 'A3_AsignadaVia' if match_label == 1 else 'A3_AsignadaViaAlternativa'

logger.info("Cargando datos desde base de datos...")
try:
    df_vv = gpd.read_postgis("SELECT * FROM trabajo.vivienda", engine, geom_col='geom', crs='EPSG:4326')
    logger.info(f"Total viviendas cargadas: {len(df_vv)}")

    df_vc = df_vv[df_vv['cvia_ine'] == 0].copy()
    logger.info(f"Viviendas con cvia_ine == 0: {len(df_vc)}")

    df_callejero_num = gpd.read_postgis("SELECT * FROM grafcan.callejero_num", engine, geom_col='geom', crs='EPSG:4326')
    logger.info(f"Portales cargados: {len(df_callejero_num)}")

    df_callejero_via = gpd.read_postgis("SELECT * FROM grafcan.callejero_via", engine, geom_col='geom', crs='EPSG:4326')
    logger.info(f"Vías cargadas: {len(df_callejero_via)}")

    df_parcela = gpd.read_postgis("SELECT * FROM catastro.parcela WHERE tipo != 'X'", engine, geom_col='geom', crs='EPSG:4326')
    logger.info(f"Parcelas cargadas: {len(df_parcela)}")

    df_poblaciones = gpd.read_postgis("SELECT * FROM age.poblaciones", engine, geom_col='geom', crs='EPSG:4326')
    logger.info(f"Poblaciones cargadas: {len(df_poblaciones)}")

except Exception as e:
    logger.error(f"Error cargando datos: {e}")
    raise

df_vc = df_vc.assign(
    cvia_ine='',
    a3_estado='',
    a3_origen='',
    a3_metodo='',
    a3_uuid='',
    cod_pob='',
    predict_proba=None,
    direccion_norm_vc='',
    direccion_norm_callejero=''
)

logger.info("Normalizando nombres de vías...")
df_vc['direccion_norm'] = df_vc.apply(lambda r: normalizar_via_nombre(r['tvia_dgc'], r['nvia_dgc']), axis=1)
df_callejero_num['direccion_norm'] = df_callejero_num.apply(lambda r: normalizar_via_nombre(r['tipovia'], r['nombrevia']), axis=1)
df_callejero_via['direccion_norm'] = df_callejero_via.apply(lambda r: normalizar_via_nombre(r['tipovia'], r['nombrevia']), axis=1)

logger.info("Corrigiendo geometrías...")
df_vc = corregir_geoms(df_vc, "Viviendas")
df_parcela = corregir_geoms(df_parcela, "Parcelas")
df_callejero_num = corregir_geoms(df_callejero_num, "Portales")
df_callejero_via = corregir_geoms(df_callejero_via, "Vías")

logger.info("Entrenando clasificador ML...")

try:
    df_test = pd.read_excel("sample_direcciones_para_match_manual.xlsx")
    
    feature_rows = []
    labels = []
    for _, row in df_test.iterrows():
        if pd.isna(row['match_manual']):
            continue
        direccion_vc = str(row['direccion_norm_vc']) if pd.notna(row['direccion_norm_vc']) else ""
        direccion_callejero = str(row['direccion_norm_callejero']) if pd.notna(row['direccion_norm_callejero']) else ""
        feats = compute_features(direccion_vc, direccion_callejero)
        feature_rows.append(feats)
        labels.append(int(row['match_manual']))

    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    logger.info("=== Evaluación del modelo ===")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    logger.info(f"\n=== Importancia de las variables ===\n{feature_importances}")

except FileNotFoundError:
    logger.warning("Archivo de validación no encontrado. Usando modelo básico.")
    clf = None

for codmun, nombre in municipios.items():
    logger.info(f"\nProcesando municipio {nombre} ({codmun})...")
    mun_id = int(codmun[-2:])

    df_vc_mun = df_vc[df_vc['cmun_ine'].astype(int) == mun_id].copy()
    df_parcela_mun = df_parcela[df_parcela['municipio'].astype(int) == mun_id].copy()
    df_callejero_num_mun = df_callejero_num[df_callejero_num['codmun'].astype(str).str[-2:].astype(int) == mun_id].copy()
    df_callejero_via_mun = df_callejero_via[df_callejero_via['codmun'].astype(str).str[-2:].astype(int) == mun_id].copy()
    df_poblaciones_mun = clean_index_right(df_poblaciones.copy())

    if df_vc_mun.empty:
        logger.info("  No hay viviendas para este municipio.")
        continue

    vc_parcela = safe_sjoin(df_vc_mun, df_parcela_mun[['refcat', 'geom']], how="left", predicate="within")

    for idx, row in tqdm(vc_parcela.iterrows(), total=vc_parcela.shape[0], desc=f"Procesando {nombre}"):
        refcat = row['refcat']
        geom_vc = row['geom']

        df_vc.at[row.name, 'direccion_norm_vc'] = row['direccion_norm']

        portal_result = process_portal_matching(row, df_callejero_num_mun, df_parcela_mun, refcat)
        
        if portal_result['found']:
            match_result = portal_result
        else:
            match_result = process_via_proximity(row, df_callejero_via_mun, geom_vc)

        a3_estado = determine_a3_estado(row, match_result, df_poblaciones_mun, df_vc.crs, clf)
        
        df_vc.at[row.name, 'a3_estado'] = a3_estado
        
        if match_result['found']:
            df_vc.at[row.name, 'cvia_ine'] = match_result['codvia']
            df_vc.at[row.name, 'a3_origen'] = match_result['origen']
            df_vc.at[row.name, 'direccion_norm_callejero'] = match_result['match_name']
            df_vc.at[row.name, 'a3_uuid'] = match_result['uuid_match']

    vc_con_pob = safe_sjoin(df_vc_mun, df_poblaciones_mun[['cunn_1', 'geom']], how="left", predicate="within", rsuffix="_pob")
    
    for idx, row in vc_con_pob.iterrows():
        if pd.notna(row['cunn_1']):
            df_vc.at[row.name, 'cod_pob'] = row['cunn_1']

df_vc['cvia_ine'] = df_vc['cvia_ine'].apply(lambda x: str(x).zfill(5))

df_vc.loc[
    (
        df_vc["direccion_norm_vc"].str.contains("NONE", na=False) |
        df_vc["direccion_norm_callejero"].str.contains("NONE", na=False)
    ) & (df_vc["a3_estado"] == "Asignada"),
    "a3_estado"
] = "AsignadaViaAlternativa"

for col in ['uuid', 'direccion_norm']:
    if col in df_vc.columns:
        df_vc = df_vc.drop(columns=[col])

logger.info("Exportando resultados...")
df_vc.to_excel("a3_asignaciones_vc.xlsx", index=False)
df_vc.to_csv("a3_asignaciones_vc.csv", index=False)

logger.info("=== RESUMEN DE ASIGNACIONES ===")
estados = ['A3_AsignadaVia', 'A3_AsignadaViaAlternativa', 'A3_AsignadaUnidadPoblacional', 'A3_Aislada']
for estado in estados:
    count = df_vc[df_vc['a3_estado'] == estado].shape[0]
    logger.info(f"{estado}: {count} registros")
logger.info(f"Total registros procesados: {len(df_vc)}")