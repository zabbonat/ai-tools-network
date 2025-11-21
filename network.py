import streamlit as st
import networkx as nx
from pyvis.network import Network
import pandas as pd
import tempfile
import os
import numpy as np

# Impostazione pagina
st.set_page_config(page_title="AI Impact Network", layout="wide")

# -----------------------------------------------------------------------------
# 1. CARICAMENTO DATI
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        # Assicuriamoci che i tipi siano corretti
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['publications'] = pd.to_numeric(df['publications'], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        return None

# -----------------------------------------------------------------------------
# 2. COSTRUZIONE DEL GRAFO
# -----------------------------------------------------------------------------
def build_graph(df_filtered, min_pubs_threshold):
    """Costruisce il grafo pesato basato sulle pubblicazioni aggregate."""
    G = nx.Graph()
    
    grouped = df_filtered.groupby(['architecture', 'field_name'])['publications'].sum().reset_index()
    grouped = grouped[grouped['publications'] >= min_pubs_threshold]
    
    if grouped.empty:
        return G

    arch_totals = grouped.groupby('architecture')['publications'].sum()
    field_totals = grouped.groupby('field_name')['publications'].sum()
    
    def get_size(val):
        return 10 + np.log1p(val) * 8

    # Nodi ARCHITETTURA
    for arch, count in arch_totals.items():
        tooltip = f"🛠️ TOOL: {arch}\n📚 Pubs (periodo): {int(count)}"
        G.add_node(arch, group='Tool', title=tooltip, value=int(count), 
                   size=get_size(count), color='#00C9FF', shape='dot')

    # Nodi FIELD
    for field, count in field_totals.items():
        tooltip = f"🌍 FIELD: {field}\n📚 Pubs (periodo): {int(count)}"
        G.add_node(field, group='Field', title=tooltip, value=int(count),
                   size=get_size(count), color='#FF6B6B', shape='dot')

    # ARCHI
    for _, row in grouped.iterrows():
        arch = row['architecture']
        field = row['field_name']
        pubs = row['publications']
        weight = 1 + np.log1p(pubs)
        title_edge = f"{arch} -> {field}: {int(pubs)} papers"
        
        G.add_edge(arch, field, title=title_edge, width=weight, color='#555555')
        
    return G

# -----------------------------------------------------------------------------
# 3. RENDERIZZAZIONE PYVIS
# -----------------------------------------------------------------------------
def render_pyvis(G):
    nt = Network(height="600px", width="100%", bgcolor="#1E1E1E", font_color="white")
    nt.from_nx(G)
    nt.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.005,
          "springLength": 200,
          "springConstant": 0.08,
          "damping": 0.4
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      },
      "nodes": { "font": { "strokeWidth": 0 } }
    }
    """)
    return nt

# -----------------------------------------------------------------------------
# INTERFACCIA STREAMLIT
# -----------------------------------------------------------------------------

st.title("🕸️ AI Architecture Diffusion Network")
st.markdown("Analisi dei flussi di adozione delle architetture AI nei campi scientifici.")

# --- CARICAMENTO ---
default_file = 'checkpoint.csv'
uploaded_file = st.sidebar.file_uploader("Carica CSV Risultati", type=['csv'])

if uploaded_file:
    df = load_data(uploaded_file)
elif os.path.exists(default_file):
    df = load_data(default_file)
else:
    st.error(f"Nessun file trovato. Carica il file '{default_file}'.")
    st.stop()

# --- SIDEBAR FILTRI ---
st.sidebar.header("🎛️ Filtri Globali")
min_year, max_year = int(df['year'].min()), int(df['year'].max())
selected_years = st.sidebar.slider("Periodo di Analisi", min_year, max_year, (min_year, max_year))
min_pubs = st.sidebar.slider("Soglia Minima Pubblicazioni", 1, 500, 50, 10)

all_archs = sorted(df['architecture'].unique())
all_fields = sorted(df['field_name'].unique())

selected_archs = st.sidebar.multiselect("Filtra Architetture", all_archs, default=[])
selected_fields = st.sidebar.multiselect("Filtra Campi", all_fields, default=[])

# --- FILTRAGGIO DATI ---
mask = (df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])
filtered_df = df[mask]
if selected_archs: filtered_df = filtered_df[filtered_df['architecture'].isin(selected_archs)]
if selected_fields: filtered_df = filtered_df[filtered_df['field_name'].isin(selected_fields)]

# --- VISUALIZZAZIONE NETWORK ---
col1, col2, col3 = st.columns(3)
col1.metric("Pubblicazioni Totali", f"{int(filtered_df['publications'].sum()):,}")

if not filtered_df.empty:
    G = build_graph(filtered_df, min_pubs)
    col2.metric("Architetture Visibili", len([n for n, a in G.nodes(data=True) if a.get('group')=='Tool']))
    col3.metric("Campi Visibili", len([n for n, a in G.nodes(data=True) if a.get('group')=='Field']))

    if G.number_of_nodes() > 0:
        nt = render_pyvis(G)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            nt.save_graph(tmp_file.name)
            with open(tmp_file.name, 'r', encoding='utf-8') as f:
                html_data = f.read()
            st.components.v1.html(html_data, height=650, scrolling=False)
            os.remove(tmp_file.name)
    else:
        st.warning("Nessuna connessione supera la soglia minima. Abbassa la 'Soglia Minima Pubblicazioni'.")
else:
    st.warning("Nessun dato nel periodo selezionato.")

# =============================================================================
# NUOVA SEZIONE: ANALISI TEMPORALE DETTAGLIATA
# =============================================================================
st.markdown("---")
st.header("📈 Analisi Temporale (Drill-down)")
st.markdown("Seleziona un'architettura specifica per vedere come si è evoluta la sua adozione anno per anno nei diversi campi.")

# 1. Selettore dedicato per l'analisi temporale
drill_arch = st.selectbox("Seleziona Architettura per il dettaglio:", all_archs)

if drill_arch:
    # Recuperiamo TUTTI i dati storici per questa architettura (ignorando parzialmente i filtri globali per dare contesto)
    # Manteniamo solo il filtro sui campi se l'utente ne ha selezionati alcuni specifici
    arch_history = df[df['architecture'] == drill_arch]
    
    if selected_fields:
        arch_history = arch_history[arch_history['field_name'].isin(selected_fields)]
        
    # Pivot Table: Righe=Anni, Colonne=Field, Valori=Pubblicazioni
    # Questo crea la struttura perfetta per st.line_chart
    chart_data = arch_history.pivot_table(
        index='year', 
        columns='field_name', 
        values='publications', 
        aggfunc='sum'
    ).fillna(0)
    
    # Assicuriamoci di coprire tutto il range di anni (per non avere buchi nel grafico)
    full_years = range(int(arch_history['year'].min()), int(arch_history['year'].max()) + 1)
    chart_data = chart_data.reindex(full_years, fill_value=0)

    # Layout grafico e statistiche
    c1, c2 = st.columns([3, 1])
    
    with c1:
        st.subheader(f"Curve di Adozione: {drill_arch}")
        # Streamlit gestisce automaticamente la legenda dei colori in base alle colonne (Fields)
        st.line_chart(chart_data)
    
    with c2:
        st.subheader("Top Fields")
        # Mostra i campi che hanno usato di più questa architettura in totale
        top_fields = arch_history.groupby('field_name')['publications'].sum().sort_values(ascending=False).head(10)
        st.dataframe(top_fields, height=400)

# --- TABELLA DATI GREZZI ---
with st.expander("📋 Visualizza Dati Completi"):
    st.dataframe(filtered_df.sort_values('publications', ascending=False), use_container_width=True)
