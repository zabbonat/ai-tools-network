import streamlit as st
import networkx as nx
from pyvis.network import Network
import pandas as pd
import tempfile
import os
import numpy as np
import altair as alt

# Page Config
st.set_page_config(page_title="AI Impact Network", layout="wide")

# -----------------------------------------------------------------------------
# 1. DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        # Ensure types are correct
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['publications'] = pd.to_numeric(df['publications'], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        return None

# -----------------------------------------------------------------------------
# 2. GRAPH CONSTRUCTION
# -----------------------------------------------------------------------------
def build_graph(df_filtered, min_pubs_threshold):
    """Builds the weighted graph based on aggregated publications."""
    G = nx.Graph()
    
    # Group data
    grouped = df_filtered.groupby(['architecture', 'field_name'])['publications'].sum().reset_index()
    grouped = grouped[grouped['publications'] >= min_pubs_threshold]
    
    if grouped.empty:
        return G

    # Calculate totals
    arch_totals = grouped.groupby('architecture')['publications'].sum()
    field_totals = grouped.groupby('field_name')['publications'].sum()
    
    def get_size(val):
        return 10 + np.log1p(val) * 8

    # Add Nodes
    for arch, count in arch_totals.items():
        tooltip = f"🛠️ ARCH: {arch}\n📚 Pubs: {int(count)}"
        G.add_node(arch, group='Tool', title=tooltip, value=int(count), 
                   size=get_size(count), color='#00C9FF', shape='dot')

    for field, count in field_totals.items():
        tooltip = f"🌍 FIELD: {field}\n📚 Pubs: {int(count)}"
        G.add_node(field, group='Field', title=tooltip, value=int(count),
                   size=get_size(count), color='#FF6B6B', shape='dot')

    # Add Edges
    for _, row in grouped.iterrows():
        arch = row['architecture']
        field = row['field_name']
        pubs = row['publications']
        weight = 1 + np.log1p(pubs)
        title_edge = f"{arch} -> {field}: {int(pubs)} papers"
        G.add_edge(arch, field, title=title_edge, width=weight, color='#555555')
        
    return G

# -----------------------------------------------------------------------------
# 3. PYVIS RENDERING
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
# STREAMLIT INTERFACE
# -----------------------------------------------------------------------------

st.title("🕸️ AI Architecture Diffusion Network")
st.markdown("Analyze how AI architectures are adopted across scientific fields.")

# --- FILE UPLOAD ---
default_file = 'checkpoint.csv'
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if uploaded_file:
    df = load_data(uploaded_file)
elif os.path.exists(default_file):
    df = load_data(default_file)
else:
    st.error("No file found.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("🎛️ Global Filters")
min_year, max_year = int(df['year'].min()), int(df['year'].max())
selected_years = st.sidebar.slider("Period", min_year, max_year, (min_year, max_year))
min_pubs = st.sidebar.slider("Min Pubs Threshold", 1, 500, 50, 10)

all_archs = sorted(df['architecture'].unique())
all_fields = sorted(df['field_name'].unique())

selected_archs = st.sidebar.multiselect("Filter Archs", all_archs, default=[])
selected_fields = st.sidebar.multiselect("Filter Fields", all_fields, default=[])

# --- NETWORK VIZ ---
mask = (df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])
filtered_df = df[mask]
if selected_archs: filtered_df = filtered_df[filtered_df['architecture'].isin(selected_archs)]
if selected_fields: filtered_df = filtered_df[filtered_df['field_name'].isin(selected_fields)]

col1, col2, col3 = st.columns(3)
col1.metric("Total Pubs", f"{int(filtered_df['publications'].sum()):,}")

if not filtered_df.empty:
    G = build_graph(filtered_df, min_pubs)
    col2.metric("Archs", len([n for n, a in G.nodes(data=True) if a.get('group')=='Tool']))
    col3.metric("Fields", len([n for n, a in G.nodes(data=True) if a.get('group')=='Field']))

    if G.number_of_nodes() > 0:
        nt = render_pyvis(G)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            nt.save_graph(tmp_file.name)
            with open(tmp_file.name, 'r', encoding='utf-8') as f:
                html_data = f.read()
            st.components.v1.html(html_data, height=650, scrolling=False)
            os.remove(tmp_file.name)
    else:
        st.warning("Graph empty. Lower threshold.")
else:
    st.warning("No data.")

# =============================================================================
# DRILL-DOWN: TEMPORAL ANALYSIS (AGGIORNATO)
# =============================================================================
st.markdown("---")
st.header("📈 Temporal Drill-down")
st.markdown("Select an architecture to analyze trends.")

drill_arch = st.selectbox("Select Architecture:", all_archs)

if drill_arch:
    # 1. Prepara i dati storici completi per l'architettura scelta
    arch_history = df[df['architecture'] == drill_arch]
    
    # Trova tutti i field disponibili per questa architettura
    available_fields = sorted(arch_history['field_name'].unique())
    
    # 2. NUOVO CONTROLLO: Multiselect locale per nascondere/mostrare field
    # Di default mostriamo tutti (o i top 10 per non intasare)
    default_fields = available_fields[:10] if len(available_fields) > 10 else available_fields
    
    c_filter, c_stats = st.columns([3, 1])
    
    with c_filter:
        st.markdown("##### 👁️ Visibility Filter (Uncheck to rescale Y-axis)")
        fields_to_show = st.multiselect(
            "Select fields to plot:", 
            options=available_fields, 
            default=default_fields
        )
    
    # Filtra i dati in base alla selezione dell'utente (questo causa il rescaling!)
    plot_data = arch_history[arch_history['field_name'].isin(fields_to_show)]
    
    # Aggregazione per il grafico
    agg_data = plot_data.groupby(['year', 'field_name'])['publications'].sum().reset_index()

    with c_filter:
        if not agg_data.empty:
            # 3. CONFIGURAZIONE ALTAIR INTERATTIVO
            # Creiamo un selettore che lega il click sulla legenda
            selection = alt.selection_point(fields=['field_name'], bind='legend')

            chart = alt.Chart(agg_data).mark_line(point=True).encode(
                x=alt.X('year:O', axis=alt.Axis(title='Year', labelAngle=0)),
                y=alt.Y('publications:Q', axis=alt.Axis(title='Publications')),
                
                # Colore condizionale: se selezionato (o nulla selezionato) = colore field, altrimenti grigio
                color=alt.Color(
                    'field_name', 
                    legend=alt.Legend(title="Fields (Click to Highlight)", orient='bottom', columns=4, labelLimit=200)
                ),
                
                # Opacità condizionale: sfuma le linee non selezionate
                opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
                
                tooltip=['year', 'field_name', 'publications']
            ).add_params(
                selection # Attiva l'interattività
            ).properties(
                height=500
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Select at least one field to see the plot.")

    with c_stats:
        st.markdown("##### 🏆 Top Adopters")
        if not arch_history.empty:
            top = arch_history.groupby('field_name')['publications'].sum().sort_values(ascending=False).head(10)
            st.dataframe(top, height=400)

# --- RAW DATA ---
with st.expander("📋 View Raw Data"):
    st.dataframe(filtered_df.sort_values('publications', ascending=False), use_container_width=True)
