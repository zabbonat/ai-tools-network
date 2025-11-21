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
    """
    Costruisce il grafo pesato basato sulle pubblicazioni aggregate.
    """
    G = nx.Graph() # O DiGraph se vuoi le frecce, ma Graph è più pulito qui
    
    # 1. Raggruppa i dati: Somma le pubblicazioni per Architettura -> Field
    # (Ignoriamo l'anno qui perché è già stato filtrato prima)
    grouped = df_filtered.groupby(['architecture', 'field_name'])['publications'].sum().reset_index()
    
    # Filtra connessioni deboli (rumore)
    grouped = grouped[grouped['publications'] >= min_pubs_threshold]
    
    # Se non ci sono dati dopo il filtro
    if grouped.empty:
        return G

    # Calcolo totali per dimensionare i nodi
    arch_totals = grouped.groupby('architecture')['publications'].sum()
    field_totals = grouped.groupby('field_name')['publications'].sum()
    
    # Scala logaritmica per la grandezza dei nodi (per evitare nodi giganti)
    def get_size(val):
        return 10 + np.log1p(val) * 8

    # Aggiunta Nodi ARCHITETTURA (Tools)
    for arch, count in arch_totals.items():
        tooltip = f"🛠️ TOOL: {arch}\n📚 Total Pubs (nel periodo): {int(count)}"
        G.add_node(
            arch,
            group='Tool',
            title=tooltip,
            value=int(count), # Per la fisica di PyVis
            size=get_size(count),
            color='#00C9FF', # Azzurro Ciano
            shape='dot'
        )

    # Aggiunta Nodi FIELD (Campi di applicazione)
    for field, count in field_totals.items():
        tooltip = f"🌍 FIELD: {field}\n📚 Total Pubs (nel periodo): {int(count)}"
        G.add_node(
            field,
            group='Field',
            title=tooltip,
            value=int(count),
            size=get_size(count),
            color='#FF6B6B', # Rosso Corallo
            shape='dot' # 'triangle' o 'diamond' per differenziare
        )

    # Aggiunta ARCHI (Collegamenti)
    for _, row in grouped.iterrows():
        arch = row['architecture']
        field = row['field_name']
        pubs = row['publications']
        
        # Lo spessore (width) dipende dalle pubblicazioni
        weight = 1 + np.log1p(pubs)
        
        title_edge = f"{arch} -> {field}: {int(pubs)} papers"
        
        G.add_edge(
            arch,
            field,
            title=title_edge,
            width=weight,
            color='#555555' # Grigio scuro
        )
        
    return G

# -----------------------------------------------------------------------------
# 3. RENDERIZZAZIONE PYVIS
# -----------------------------------------------------------------------------
def render_pyvis(G):
    nt = Network(
        height="700px", 
        width="100%", 
        bgcolor="#1E1E1E", # Sfondo scuro stile "Dark Mode"
        font_color="white"
    )
    nt.from_nx(G)
    
    # Configurazione Fisica: Forza di repulsione per distanziare i nodi
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
      "nodes": {
          "font": { "strokeWidth": 0 }
      }
    }
    """)
    return nt

# -----------------------------------------------------------------------------
# INTERFACCIA STREAMLIT
# -----------------------------------------------------------------------------

st.title("🕸️ AI Architecture Diffusion Network")
st.markdown("""
Analizza come le architetture AI (Nodi Azzurri) vengono adottate 
nei diversi campi scientifici (Nodi Rossi) basandosi sul volume di pubblicazioni.
""")

# 1. Carica file
default_file = 'checkpoint.csv'
uploaded_file = st.sidebar.file_uploader("Carica CSV Risultati", type=['csv'])

if uploaded_file:
    df = load_data(uploaded_file)
elif os.path.exists(default_file):
    df = load_data(default_file)
    st.sidebar.success(f"File predefinito caricato: {len(df)} righe")
else:
    st.error(f"Nessun file trovato. Carica il file '{default_file}' generato dallo script precedente.")
    st.stop()

# 2. Controlli Sidebar
st.sidebar.header("🎛️ Filtri Temporali e Metriche")

# Slider Anni
min_year = int(df['year'].min())
max_year = int(df['year'].max())
selected_years = st.sidebar.slider(
    "Periodo di Analisi",
    min_year, max_year, (min_year, max_year)
)

# Filtro Soglia (importante per non avere un "hairball" graph)
min_pubs = st.sidebar.slider(
    "Soglia Minima Pubblicazioni (Filtra rumore)",
    min_value=1, max_value=500, value=50, step=10,
    help="Mostra solo le connessioni con almeno X pubblicazioni in totale nel periodo selezionato."
)

# Filtro opzionale per Architettura specifica
all_archs = sorted(df['architecture'].unique())
selected_archs = st.sidebar.multiselect(
    "Filtra per Architetture (Lascia vuoto per tutte)",
    all_archs,
    default=[] # Default vuoto = tutte
)

# Filtro opzionale per Field specifico
all_fields = sorted(df['field_name'].unique())
selected_fields = st.sidebar.multiselect(
    "Filtra per Campi (Lascia vuoto per tutti)",
    all_fields,
    default=[]
)

# 3. Elaborazione Dati
# Filtra per anni
mask = (df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])
filtered_df = df[mask]

# Filtra per selezione utente (se presente)
if selected_archs:
    filtered_df = filtered_df[filtered_df['architecture'].isin(selected_archs)]
if selected_fields:
    filtered_df = filtered_df[filtered_df['field_name'].isin(selected_fields)]

# 4. Visualizzazione
if filtered_df.empty:
    st.warning("Nessun dato trovato con i filtri attuali.")
else:
    # Costruisci grafo
    G = build_graph(filtered_df, min_pubs)
    
    # Statistiche rapide sopra il grafo
    col1, col2, col3 = st.columns(3)
    total_pubs_period = filtered_df['publications'].sum()
    col1.metric("Pubblicazioni Totali (Periodo)", f"{int(total_pubs_period):,}")
    col2.metric("Architetture Attive", len([n for n, attr in G.nodes(data=True) if attr.get('group') == 'Tool']))
    col3.metric("Campi Coinvolti", len([n for n, attr in G.nodes(data=True) if attr.get('group') == 'Field']))

    if G.number_of_nodes() > 0:
        nt = render_pyvis(G)
        
        # Hack per renderizzare PyVis in Streamlit
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            nt.save_graph(tmp_file.name)
            with open(tmp_file.name, 'r', encoding='utf-8') as f:
                html_data = f.read()
            st.components.v1.html(html_data, height=750, scrolling=False)
            os.remove(tmp_file.name)
    else:
        st.warning("Il grafo è vuoto. Prova ad abbassare la 'Soglia Minima Pubblicazioni'.")

# 5. Tabella Dati Sottostante
with st.expander("📋 Visualizza Dati Aggregati"):
    grouped_display = filtered_df.groupby(['architecture', 'field_name'])['publications'].sum().reset_index().sort_values('publications', ascending=False)
    st.dataframe(grouped_display, use_container_width=True)