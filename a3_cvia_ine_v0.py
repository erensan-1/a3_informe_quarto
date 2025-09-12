# === LIBRERÍAS NECESARIAS ===
import os
import pandas as pd
import geopandas as gpd
import psycopg2
import unicodedata
import re
import pandas as pd
from sqlalchemy import create_engine
from shapely.geometry import Point
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

# Para embeddings
from sentence_transformers import SentenceTransformer

# === CONEXIÓN A LA BASE DE DATOS ===
DB_URI = "postgresql+psycopg2://erensan:5eP79ECt@192.168.21.226:5432/padron_online"
engine = create_engine(DB_URI)

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
    'CZDA': 'CALZADA', 'CZADA': 'CALZADA', 'VREDA': 'VEREDA'
}

# === EQUIVALENCIAS DE VÍAS ===
equivalencias_vias = {
    ('CALLE', 'SAN JOSE LA CISNERA'): ('CALLE', 'SAN JOSE'),
    ('CALLE', 'JOSE GREGORIO YANES DORTA'): ('CARRETERA', 'DE LA VERA BAJA')
}

# === FUNCIONES ===
def limpiar_texto(texto):
    if not isinstance(texto, str):
        texto = str(texto)
    texto = texto.upper().strip()
    texto = unicodedata.normalize('NFKD', texto)
    texto = ''.join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r'[^A-Z0-9 ]', '', texto)
    return texto

def normalizar_direccion(cmun, tvia, nvia):
    try:
        cmun_norm = str(int(cmun)).zfill(3)
    except (ValueError, TypeError):
        cmun_norm = limpiar_texto(cmun).zfill(3)

    try:
        tvia_str = str(tvia).strip().upper()
        tvia_raw = SIGLAS_VIA.get(tvia_str, tvia_str)
    except:
        tvia_raw = str(tvia)

    tvia_norm = limpiar_texto(tvia_raw)
    nvia_norm = limpiar_texto(nvia)

    key_eq = (tvia_norm, nvia_norm)
    if key_eq in equivalencias_vias:
        nuevo_tvia, nuevo_nvia = equivalencias_vias[key_eq]
        tvia_norm = limpiar_texto(nuevo_tvia)
        nvia_norm = limpiar_texto(nuevo_nvia)

    return f"{cmun_norm} {tvia_norm} {nvia_norm}"

def asignar_cvia_ine_con_direccion(row):
    cmun = row['cmun_norm']
    direccion = row['direccion_vv']
    diccionario_local = cce_por_municipio.get(cmun, {})
    cvia_encontrado = diccionario_local.get(direccion, row['cvia_ine'])
    direccion_cce = diccionario_local.get(direccion, direccion)  # ahora siempre devuelve algo
    return pd.Series([cvia_encontrado, direccion_cce])

def limpiar_texto_sufijos(texto):
    texto = limpiar_texto(texto)
    tokens = texto.split()
    tokens = [t for t in tokens if t not in SUFIJOS_COMUNES]
    return ' '.join(tokens)

# === FUNCIÓN DE ASIGNACIÓN CON DIRECCION_CCE ===
def asignar_cvia(row):
    cmun = row['cmun_norm']
    direccion = row['direccion_vv']
    diccionario_local = cce_por_municipio.get(cmun, {})

    # CVIA: si hay match exacto, usamos el valor del diccionario; si no, mantenemos cvia_ine
    cvia_asignado = diccionario_local.get(direccion, row['cvia_ine'])
    
    # DIRECCION_CCE: solo asignar si hay match exacto, si no, dejar vacío
    if direccion in diccionario_local:
        direccion_cce = direccion
    else:
        direccion_cce = ''  # mantener vacío si no hay match exacto

    return pd.Series([cvia_asignado, direccion_cce])

SUFIJOS_COMUNES = ['EL', 'LA', 'LOS', 'LAS']

# === CONSULTAS A LA BBDD ===
query_vv = """
SELECT *, ST_SRID(geom) AS srid
FROM trabajo.vivienda
WHERE cvia_ine = '0' AND (tvia_dgc IS NOT NULL OR nvia_dgc IS NOT NULL)
"""
query_cce_via = "SELECT cmun, cvia, tvia_var, nvia_var FROM ine.cce_via"

df_vv = pd.read_sql_query(query_vv, engine)
df_cce_via = pd.read_sql_query(query_cce_via, engine)

df_vv['cvia_ine'] = df_vv['cvia_ine'].astype(str)

# === NORMALIZAR DIRECCIONES ===
df_vv['direccion_vv'] = df_vv.apply(lambda r: normalizar_direccion(r['cmun_ine'], r['tvia_dgc'], r['nvia_dgc']), axis=1)
df_cce_via['direccion_cce'] = df_cce_via.apply(lambda r: normalizar_direccion(r['cmun'], r['tvia_var'], r['nvia_var']), axis=1)
df_vv['cmun_norm'] = df_vv['cmun_ine'].apply(lambda x: str(int(x)).zfill(3) if pd.notnull(x) else '')
df_cce_via['cmun_norm'] = df_cce_via['cmun'].apply(lambda x: str(int(x)).zfill(3) if pd.notnull(x) else '')

# === CREAR DICCIONARIO POR MUNICIPIO ===
cce_por_municipio = df_cce_via.groupby('cmun_norm').apply(lambda x: dict(zip(x['direccion_cce'], x['cvia']))).to_dict()

# === MATCH EXACTO ===
df_vv['cvia_ine_original'] = df_vv['cvia_ine']
df_vv[['cvia_ine', 'direccion_cce_via']] = df_vv.apply(asignar_cvia_ine_con_direccion, axis=1)
df_vv['metodo'] = df_vv.apply(lambda r: 'match exacto' if r['cvia_ine_original'] == '0' and r['cvia_ine'] != '0' else None, axis=1)

# === CARGAR MATCH MANUAL ===
script_dir = os.path.dirname(os.path.abspath(__file__))
ruta_match_manual = os.path.join(script_dir,"a3_match_manual.xlsx")
df_match_manual = pd.read_excel(ruta_match_manual)

columnas_clave = ['cmun_ine', 'tvia_dgc', 'nvia_dgc', 'direccion_vv']
df_match_manual = df_match_manual[columnas_clave + ['match_manual']]

df_vv = df_vv.drop(columns=['match_manual'], errors='ignore')
df_vv = df_vv.merge(df_match_manual, on=columnas_clave, how='left')

# === FUZZY MATCHING OPTIMIZADO POR MUNICIPIO ===
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === FUZZY MATCHING OPTIMIZADO POR MUNICIPIO CON SCORE_COSENO ===
# Preparar embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings_cache = {}

# Filtramos solo filas sin match
sin_match = df_vv[df_vv['cvia_ine'] == '0'].copy()

# Para cache TF-IDF por municipio
tfidf_cache = {}

for idx, row in sin_match.iterrows():
    cmun = str(int(row['cmun_ine'])).zfill(3)
    direccion_vv = row['direccion_vv']

    if cmun in cce_por_municipio:
        direcciones_cce = list(cce_por_municipio[cmun].keys())
        
        # Preselección top 5 con token_sort_ratio para evaluar mejor
        matches = process.extract(
            direccion_vv, direcciones_cce, scorer=fuzz.token_sort_ratio, limit=5
        )
        
        mejor_puntaje = -1
        mejor_direccion = None
        mejor_scores = {}

        # Preparar TF-IDF para municipio si no está en cache
        if cmun not in tfidf_cache:
            vect = TfidfVectorizer().fit(direcciones_cce + [direccion_vv])
            tfidf_cache[cmun] = {
                'vectorizer': vect,
                'tfidf_matrix': vect.transform(direcciones_cce)
            }
        else:
            vect = tfidf_cache[cmun]['vectorizer']

        # TF-IDF vector para direccion_vv
        tfidf_vv = vect.transform([direccion_vv])

        for candidate, score_token_sort, _ in matches:
            score_ratio = fuzz.ratio(direccion_vv, candidate)
            score_partial = fuzz.partial_ratio(direccion_vv, candidate)
            score_token_set = fuzz.token_set_ratio(direccion_vv, candidate)

            # Embeddings
            if direccion_vv not in embeddings_cache:
                embeddings_cache[direccion_vv] = model.encode(direccion_vv)
            emb_vv = embeddings_cache[direccion_vv]

            if candidate not in embeddings_cache:
                embeddings_cache[candidate] = model.encode(candidate)
            emb_cce = embeddings_cache[candidate]

            score_embedding = np.dot(emb_vv, emb_cce) / (np.linalg.norm(emb_vv) * np.linalg.norm(emb_cce)) * 100

            # Score coseno TF-IDF
            idx_candidate = direcciones_cce.index(candidate)
            tfidf_candidate = tfidf_cache[cmun]['tfidf_matrix'][idx_candidate]
            score_coseno = cosine_similarity(tfidf_vv, tfidf_candidate)[0][0] * 100

            # Promedio simple para elegir el mejor candidato
            score_promedio = np.mean([
                score_token_sort, score_ratio, score_partial, score_token_set, score_embedding, score_coseno
            ])

            if score_promedio > mejor_puntaje:
                mejor_puntaje = score_promedio
                mejor_direccion = candidate
                mejor_scores = {
                    'score_token_sort': score_token_sort,
                    'score_ratio': score_ratio,
                    'score_partial': score_partial,
                    'score_token_set': score_token_set,
                    'score_embedding': score_embedding,
                    'score_coseno': score_coseno
                }

        # Guardar resultados en df_vv
        df_vv.at[idx, 'direccion_cce_via_tmp'] = mejor_direccion
        for key, val in mejor_scores.items():
            df_vv.at[idx, key] = val

# === ENTRENAR MODELO Y CALCULAR UMBRAL ===
df_eval = df_vv[df_vv['match_manual'].isin([0, 1])].copy()

features = ['score_token_sort', 'score_coseno', 'score_ratio', 'score_partial', 'score_token_set', 'score_embedding']
X = df_eval[features].fillna(0).values
y = df_eval['match_manual'].astype(int).values

clf = LogisticRegression(max_iter=1000).fit(X, y)
probs = clf.predict_proba(X)[:, 1]

prec, rec, thr = precision_recall_curve(y, probs)
f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
idx_thr = np.nanargmax(f1s)
best_threshold = 0.885
#best_threshold = thr[idx_thr]
print(f"Mejor umbral (LR): {best_threshold:.4f}")

df_eval['prob_match'] = probs
df_eval['pred_match'] = (probs >= best_threshold).astype(int)

print(df_eval.groupby(['match_manual', 'pred_match']).size())

# === ASIGNAR MATCHES SEGÚN MODELO Y UMBRAL ===
df_vv.loc[df_vv['match_manual'].isna(), 'pred_match'] = None
mask_sin_manual = df_vv['match_manual'].isna() & (df_vv['score_token_sort'].notnull())
df_vv.loc[mask_sin_manual, 'prob_match'] = clf.predict_proba(df_vv.loc[mask_sin_manual, features].fillna(0).values)[:, 1]
df_vv.loc[mask_sin_manual, 'pred_match'] = (df_vv.loc[mask_sin_manual, 'prob_match'] >= best_threshold).astype(int)

mask_fuzzy_asignar = (df_vv['pred_match'] == 1) & (df_vv['cvia_ine'] == '0') & (df_vv['match_manual'].isna())
df_vv.loc[mask_fuzzy_asignar, 'cvia_ine'] = df_vv.loc[mask_fuzzy_asignar, 'direccion_cce_via_tmp'].map(
    lambda x: cce_por_municipio.get(x.split(' ')[0], {}).get(x, '0') if isinstance(x, str) else '0'
)

df_vv.loc[mask_fuzzy_asignar, 'direccion_cce_via'] = df_vv.loc[mask_fuzzy_asignar, 'direccion_cce_via_tmp']

df_vv.loc[(df_vv['pred_match'] == 1), 'metodo'] = df_vv.loc[(df_vv['pred_match'] == 1), 'metodo'].fillna('fuzzy + embedding + coseno')


'''# === GRAFICO ===
plt.figure(figsize=(10, 6))
plt.plot(rec, prec, label="Curva Precision-Recall")
plt.scatter(rec[idx_thr], prec[idx_thr], c='red', label=f'Umbra óptimo {best_threshold:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precisión')
plt.legend()
plt.grid()
plt.title('Precision-Recall Curve para Clasificador de Direcciones')
plt.show()'''

# === RESULTADOS ===
print(df_vv['metodo'].value_counts())
print("Número de viviendas sin CVIA asignado:", (df_vv['cvia_ine'] == '0').sum())
print("Número de viviendas con CVIA asignado:", (df_vv['cvia_ine'] != '0').sum())

# SEGUNDO FUZZY MATCHING 
from rapidfuzz import process, fuzz
import math
import numpy as np

# --- Configuración del umbral (estrictamente > 91.89)
umbral = 91.89
score_cutoff = np.nextafter(umbral, math.inf)  # garantiza "estrictamente mayor" que 91.89

# --- Máscara de pendientes
mask_pendientes = df_vv['cvia_ine'] == '0'

# --- Limpieza previa (ya la tienes)
df_vv.loc[mask_pendientes, 'direccion_clean'] = (
    df_vv.loc[mask_pendientes, 'direccion_vv'].apply(limpiar_texto_sufijos)
)
df_cce_via['direccion_clean'] = df_cce_via['direccion_cce'].apply(limpiar_texto_sufijos)

# --- Construir diccionario por municipio, filtrando vacíos
def _valid_str(s):
    return isinstance(s, str) and s.strip() != ""

df_cce_via['_ok'] = df_cce_via['direccion_clean'].apply(_valid_str)
cce_por_municipio_clean = (
    df_cce_via[df_cce_via['_ok']]
    .groupby('cmun_norm')
    .apply(lambda x: dict(zip(x['direccion_clean'], x['cvia'])))
    .to_dict()
)

registros_asignados = 0

for idx, row in df_vv[mask_pendientes].iterrows():
    cmun = row['cmun_norm']
    direccion = row['direccion_clean']

    # saltar si query vacía o no-string
    if not _valid_str(direccion):
        continue

    if cmun not in cce_por_municipio_clean:
        continue

    dict_cce = cce_por_municipio_clean[cmun]

    # candidatos no vacíos
    candidatos = [c for c in dict_cce.keys() if _valid_str(c)]
    if not candidatos:
        continue

    # Usar el mismo scorer que tu métrica y evitar reprocesado
    match = process.extractOne(
        direccion,
        candidatos,
        scorer=fuzz.token_set_ratio,
        processor=None,          # ya limpiaste tú
        score_cutoff=score_cutoff  # NO devolverá nada si <= 91.89
    )

    if match:  # si pasa el cutoff, ya es > 91.89
        direccion_match, score = match[0], float(match[1])
        cvia_match = dict_cce[direccion_match]

        df_vv.at[idx, 'cvia_ine'] = cvia_match
        df_vv.at[idx, 'direccion_cce_via'] = direccion_match
        df_vv.at[idx, 'score_token_set'] = score  # opcional: guarda el score usado
        df_vv.at[idx, 'metodo'] = (
            df_vv.at[idx, 'metodo'] if pd.notnull(df_vv.at[idx, 'metodo']) else 'fuzzy secundario'
        )
        registros_asignados += 1

# Informe
pendientes_iniciales = mask_pendientes.sum()
pendientes_restantes = (df_vv['cvia_ine'] == '0').sum()
print(f"Pendientes que entraron al segundo fuzzy: {pendientes_iniciales}")
print(f"Registros recuperados por segundo fuzzy (> {umbral}): {registros_asignados}")
print(f"Registros que siguen pendientes: {pendientes_restantes}")

# === LIMPIEZA ===
df_vv.drop(columns=['direccion_cce_via_tmp', 'cvia_ine_original', 'srid', 'cmun_norm'], errors='ignore', inplace=True)

# === EXPORTACIONES ===
output_path_match_exacto = os.path.join(script_dir, "a3_match_exacto.xlsx")
df_vv[df_vv['metodo'] == 'match exacto'].to_excel(output_path_match_exacto, index=False)
print(f"Archivo Excel (match exacto) exportado a: {output_path_match_exacto}")

output_path_fuzzy = os.path.join(script_dir, "a3_fuzzy.xlsx")
df_vv[df_vv['metodo'] == 'fuzzy'].to_excel(output_path_fuzzy, index=False)
print(f"Archivo Excel (fuzzy) exportado a: {output_path_fuzzy}")

output_completo = os.path.join(script_dir,"a3_completo.xlsx")
df_vv.to_excel(output_completo, index=False)
print(f"Archivo Excel completo exportado a: {output_completo}")

output_completo = os.path.join(script_dir,"a3_completo.csv")
df_vv.to_csv(output_completo, index=False)
print(f"Archivo Excel completo exportado a: {output_completo}")