# === LIBRERÍAS NECESARIAS ===
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# === CONEXIÓN A LA BASE DE DATOS ===
print("Conectando a la base de datos...")
DB_URI = "postgresql+psycopg2://erensan:5eP79ECt@192.168.21.226:5432/padron_online"
engine = create_engine(DB_URI)

# === CARGA DE DATOS ===
df_vv = gpd.read_postgis("SELECT * FROM trabajo.vivienda", engine, geom_col='geom', crs='EPSG:4326')
print("Total viviendas cargadas:", len(df_vv))

df_vc = df_vv[df_vv['cvia_ine'] == 0].copy()
print("Viviendas con cvia_ine == 0:", len(df_vc))

df_callejero_num = gpd.read_postgis("SELECT * FROM grafcan.callejero_num", engine, geom_col='geom', crs='EPSG:4326')
print("Portales cargados:", len(df_callejero_num))

df_callejero_via = gpd.read_postgis("SELECT * FROM grafcan.callejero_via", engine, geom_col='geom', crs='EPSG:4326')
print("Vías cargadas:", len(df_callejero_via))

df_parcela = gpd.read_postgis("SELECT * FROM catastro.parcela WHERE tipo != 'X'", engine, geom_col='geom', crs='EPSG:4326')
print("Parcelas cargadas (tipo distinto de 'X'):", len(df_parcela))

# === INICIALIZAR CAMPOS AUXILIARES ===
df_vc['cvia_ine'] = ''
df_vc['a3_estado'] = ''
df_vc['a3_origen'] = ''
df_vc['a3_metodo'] = ''
df_vc['a3_uuid'] = ''
df_vc['predict_proba'] = None
df_vc['direccion_norm_vc'] = ''
df_vc['direccion_norm_callejero'] = ''

# === DICCIONARIO DE SIGLAS DE VÍA ===
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

# === DICCIONARIO DE MUNICIPIOS ===
municipios = {
    "38004": "Arafo",
    "38005": "Arico",
    "38010": "Buenavista del Norte",
    "38012": "Fasnia",
    "38015": "Garachico",
    "38018": "Guancha, La",
    "38025": "Matanza de Acentejo, La",
    "38032": "Rosario, El",
    "38034": "San Juan de la Rambla",
    "38039": "Santa Úrsula",
    "38040": "Santiago del Teide",
    "38041": "Sauzal, El",
    "38042": "Silos, Los",
    "38044": "Tanque, El",
    "38046": "Tegueste",
    "38051": "Victoria de Acentejo, La",
    "38052": "Vilaflor de Chasna"
}

# === FUNCIONES AUXILIARES ===
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
    print(f" - {nombre} inicial: {len(gdf)}")
    gdf['geom'] = gdf['geom'].apply(lambda g: make_valid(g) if g is not None else None)
    gdf = gdf[gdf.is_valid & ~gdf.is_empty]
    print(f" - {nombre} válidas: {len(gdf)}")
    return gdf


# === NORMALIZACIÓN ===
print("Normalizando nombres de vías...")
df_vc['direccion_norm'] = df_vc.apply(lambda r: normalizar_via_nombre(r['tvia_dgc'], r['nvia_dgc']), axis=1)
df_callejero_num['direccion_norm'] = df_callejero_num.apply(lambda r: normalizar_via_nombre(r['tipovia'], r['nombrevia']), axis=1)
df_callejero_via['direccion_norm'] = df_callejero_via.apply(lambda r: normalizar_via_nombre(r['tipovia'], r['nombrevia']), axis=1)

print("Corrigiendo geometrías...")
df_vc = corregir_geoms(df_vc, "Viviendas")
df_parcela = corregir_geoms(df_parcela, "Parcelas")
df_callejero_num = corregir_geoms(df_callejero_num, "Portales")
df_callejero_via = corregir_geoms(df_callejero_via, "Vías")

# === ENTRENAR MODELO RF CON VALIDACIÓN ===
print("Entrenando clasificador ML...")

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

# Cargar excel con validación manual
df_test = pd.read_excel("sample_direcciones_para_match_manual.xlsx")

feature_rows = []
labels = []
for _, row in df_test.iterrows():
    if pd.isna(row['match_manual']):
        continue
    # Asegurarse de que las direcciones no sean nulas
    direccion_vc = str(row['direccion_norm_vc']) if pd.notna(row['direccion_norm_vc']) else ""
    direccion_callejero = str(row['direccion_norm_callejero']) if pd.notna(row['direccion_norm_callejero']) else ""
    feats = compute_features(direccion_vc, direccion_callejero)
    feature_rows.append(feats)
    labels.append(int(row['match_manual']))

X = pd.DataFrame(feature_rows)
y = pd.Series(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
clf.fit(X_train, y_train)

# === Evaluación del modelo ===
print("=== Evaluación del modelo ===")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# === Importancia de cada variable ===
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\n=== Importancia de las variables ===")
print(feature_importances)

# Opcional: gráfico de importancia
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.barh(feature_importances['feature'], feature_importances['importance'])
plt.gca().invert_yaxis()  # Mostrar la más importante arriba
plt.title("Importancia de cada variable en el Random Forest")
plt.xlabel("Importancia")
plt.show()

def filtro_heuristico(vc, callejero):
    tsr_set = fuzz.token_set_ratio(vc, callejero)
    wr = fuzz.WRatio(vc, callejero)
    if tsr_set >= 95:
        return 1
    elif wr < 60:
        return 0
    else:
        return None

def decidir_match(vc, callejero, clf=clf):
    h = filtro_heuristico(vc, callejero)
    if h is not None:
        return h
    feats = pd.DataFrame([compute_features(vc, callejero)])
    return int(clf.predict(feats)[0])

# === PROCESAR MUNICIPIO POR MUNICIPIO ===
for codmun, nombre in municipios.items():
    print(f"\nProcesando municipio {nombre} ({codmun})...")
    mun_id = int(codmun[-2:])

    df_vc_mun = df_vc[df_vc['cmun_ine'].astype(int) == mun_id].copy()
    df_parcela_mun = df_parcela[df_parcela['municipio'].astype(int) == mun_id].copy()
    df_callejero_num_mun = df_callejero_num[df_callejero_num['codmun'].astype(str).str[-2:].astype(int) == mun_id].copy()
    df_callejero_via_mun = df_callejero_via[df_callejero_via['codmun'].astype(str).str[-2:].astype(int) == mun_id].copy()

    if df_vc_mun.empty:
        print("  No hay viviendas para este municipio.")
        continue

    vc_parcela = gpd.sjoin(
        df_vc_mun,
        df_parcela_mun[['refcat', 'geom']],
        how="left",
        predicate="within"
    )

    for idx, row in tqdm(vc_parcela.iterrows(), total=vc_parcela.shape[0], desc=f"Procesando {nombre}"):
        refcat = row['refcat']
        geom_vc = row['geom']
        df_vc.at[idx, 'direccion_norm_vc'] = row['direccion_norm']

        # PRIORIDAD 1: PORTALES EN LA PARCELA
        if pd.notna(refcat):
            parcela_geom = df_parcela_mun.loc[df_parcela_mun['refcat'] == refcat, 'geom'].iloc[0]
            cn_candidatos = df_callejero_num_mun[df_callejero_num_mun.within(parcela_geom)]
        else:
            cn_candidatos = gpd.GeoDataFrame(columns=df_callejero_num_mun.columns, geometry='geom', crs=df_callejero_num_mun.crs)

        if not cn_candidatos.empty:
            matches = process.extractOne(row['direccion_norm'], cn_candidatos['direccion_norm'].tolist(), scorer=fuzz.WRatio)
            if matches:
                match_name = matches[0]
                feats = pd.DataFrame([compute_features(row['direccion_norm'], match_name)])
                proba = clf.predict_proba(feats)[0][1]  # probabilidad de ser match
                df_vc.at[idx, 'predict_proba'] = proba

                match_label = decidir_match(row['direccion_norm'], match_name, clf)
                codvia = cn_candidatos.loc[cn_candidatos['direccion_norm'] == match_name, 'codvia'].iloc[0]
                uuid_match = cn_candidatos.loc[cn_candidatos['direccion_norm'] == match_name, 'uuid'].iloc[0]

                df_vc.at[idx, 'cvia_ine'] = codvia
                df_vc.at[idx, 'a3_origen'] = 'callejero_num'
                df_vc.at[idx, 'direccion_norm_callejero'] = match_name
                df_vc.at[idx, 'a3_estado'] = 'Asignada' if match_label == 1 else 'AsignadaViaAlternativa'
                df_vc.at[idx, 'a3_metodo'] = 'heurística' if filtro_heuristico(row['direccion_norm'], match_name) is not None else 'random_forest'
                df_vc.at[idx, 'a3_uuid'] = uuid_match

        else:
            # PRIORIDAD 2: VÍAS CERCANAS
            buffer_geom = geom_vc.buffer(100)
            vias_cercanas = df_callejero_via_mun[df_callejero_via_mun.intersects(buffer_geom)]
            if not vias_cercanas.empty:
                matches = process.extractOne(row['direccion_norm'], vias_cercanas['direccion_norm'].tolist(), scorer=fuzz.WRatio)
                if matches:
                    match_name = matches[0]
                    feats = pd.DataFrame([compute_features(row['direccion_norm'], match_name)])
                    proba = clf.predict_proba(feats)[0][1]
                    df_vc.at[idx, 'predict_proba'] = proba

                    match_label = decidir_match(row['direccion_norm'], match_name, clf)
                    codvia = vias_cercanas.loc[vias_cercanas['direccion_norm'] == match_name, 'codvia'].iloc[0]
                    uuid_match = vias_cercanas.loc[vias_cercanas['direccion_norm'] == match_name, 'uuid'].iloc[0]

                    df_vc.at[idx, 'cvia_ine'] = codvia
                    df_vc.at[idx, 'a3_estado'] = 'Asignada' if match_label == 1 else 'AsignadaViaAlternativa'
                    df_vc.at[idx, 'a3_origen'] = 'callejero_via'
                    df_vc.at[idx, 'direccion_norm_callejero'] = match_name
                    df_vc.at[idx, 'a3_metodo'] = 'heurística' if filtro_heuristico(row['direccion_norm'], match_name) is not None else 'random_forest'
                    df_vc.at[idx, 'a3_uuid'] = uuid_match

            else:
                # PRIORIDAD 3: VIVIENDA AISLADA
                gdf_row = gpd.GeoDataFrame([row], geometry='geom', crs=df_vc.crs)
                nearest = gpd.sjoin_nearest(
                    gdf_row,
                    df_callejero_via_mun[['geom','codvia','uuid']],
                    how='left',
                    rsuffix='_right'
                )

                # Detectar columnas renombradas por sjoin_nearest
                uuid_col = 'uuid' if 'uuid' in nearest.columns else 'uuid_right' if 'uuid_right' in nearest.columns else 'uuid__right'
                codvia_col = 'codvia' if 'codvia' in nearest.columns else 'codvia_right' if 'codvia_right' in nearest.columns else 'codvia__right'

                if not nearest.empty:
                    df_vc.at[idx, 'cvia_ine'] = nearest[codvia_col].iloc[0]
                    df_vc.at[idx, 'a3_uuid'] = nearest[uuid_col].iloc[0]

                df_vc.at[idx, 'a3_estado'] = 'ViviendaAisladaDeViario'
                df_vc.at[idx, 'a3_origen'] = 'callejero_via'
                df_vc.at[idx, 'direccion_norm_callejero'] = ''
                df_vc.at[idx, 'a3_metodo'] = 'random_forest'
                

# CORREGIR FORMATO CODVIA
df_vc['cvia_ine'] = df_vc['cvia_ine'].apply(lambda x: str(x).zfill(5))

df_vc.loc[
    (
        df_vc["direccion_norm_vc"].str.contains("NONE", na=False) |
        df_vc["direccion_norm_callejero"].str.contains("NONE", na=False)
    ) & (df_vc["a3_estado"] == "Asignada"),
    "a3_estado"
] = "AsignadaViaAlternativa"

# EXPORTAR
for col in ['uuid', 'direccion_norm']:
    if col in df_vc.columns:
        df_vc = df_vc.drop(columns=[col])

print("Exportando resultados a Excel y CSV...")
df_vc.to_excel("a3_asignaciones_vc.xlsx", index=False)
df_vc.to_csv("a3_asignaciones_vc.csv", index=False)

# RESUMEN
print("=== RESUMEN DE ASIGNACIONES ===")
for estado in ['Asignada','AsignadaViaAlternativa','ViviendaAisladaDeViario']:
    count = df_vc[df_vc['a3_estado'] == estado].shape[0]
    print(f"{estado}: {count} registros")
print(f"Total registros procesados: {len(df_vc)}")
