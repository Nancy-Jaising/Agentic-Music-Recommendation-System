import streamlit as st
import pandas as pd
import sys, os
from urllib.parse import quote

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommender.preprocessor   import load_and_clean
from recommender.feature_engine import build_feature_matrix, transform_user_vector
from recommender.similarity     import build_knn_model, get_recommendations
from recommender.clustering     import build_clusters, get_user_cluster, filter_by_cluster
from agent.intent_parser        import parse_intent
from agent.explainer            import explain_all

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# ── Theme colors ──────────────────────────────────────────────
COLOR_OPTIONS = {
    "🟢 Spotify Green": "#1DB954",
    "🔵 Ocean Blue":    "#2196F3",
    "🟠 Sunset Orange": "#FF6B35",
    "🟣 Royal Purple":  "#9C27B0",
    "🩷 Hot Pink":      "#E91E8C",
    "🔴 Cherry Red":    "#F44336",
    "🩵 Cyan":          "#00BCD4",
    "🟡 Golden":        "#FFC107",
}

# ── Pipeline loader ───────────────────────────────────────────
@st.cache_resource(show_spinner="🎵 Loading music pipeline...")
def load_pipeline():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, 'data', 'spotify_tracks.csv')
    
    df = load_and_clean(path)
        
    X_scaled, scaler, feats = build_feature_matrix(df)
    knn_model               = build_knn_model(X_scaled, n_neighbors=20)
    df_clustered, kmeans, _ = build_clusters(X_scaled, df)
    
    return df, df_clustered, X_scaled, scaler, knn_model, kmeans

# ── URL generators ────────────────────────────────────────────
def spotify_url(track_name, artist):
    q = quote(f"{track_name} {artist}")
    return f"https://open.spotify.com/search/{q}"

def youtube_url(track_name, artist):
    q = quote(f"{track_name} {artist} official")
    return f"https://www.youtube.com/results?search_query={q}"

# ── Feature bar ───────────────────────────────────────────────
def feat_bar(label, value, accent, max_val=1.0):
    pct = min(int((value / max_val) * 100), 100)
    display = round(value, 1) if value > 10 else round(value, 2)
    return f"""
    <div style="margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;
                  font-size:0.78rem;margin-bottom:3px">
        <span style="color:#aaa">{label}</span>
        <span style="color:#fff;font-weight:600">{display}</span>
      </div>
      <div style="background:#2a2a2a;border-radius:4px;height:6px;width:100%">
        <div style="background:{accent};border-radius:4px;
                    height:6px;width:{pct}%"></div>
      </div>
    </div>"""

def feat_bar_light(label, value, accent, max_val=1.0):
    pct = min(int((value / max_val) * 100), 100)
    display = round(value, 1) if value > 10 else round(value, 2)
    return f"""
    <div style="margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;
                  font-size:0.78rem;margin-bottom:3px">
        <span style="color:#555">{label}</span>
        <span style="color:#111;font-weight:600">{display}</span>
      </div>
      <div style="background:#ddd;border-radius:4px;height:6px;width:100%">
        <div style="background:{accent};border-radius:4px;
                    height:6px;width:{pct}%"></div>
      </div>
    </div>"""

# ════════════════════════════════════════════════════════════════
# SIDEBAR CONTROL
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎨 Theme")
    dark_mode    = st.toggle("Dark Mode", value=True)
    accent_label = st.selectbox("Accent Color", list(COLOR_OPTIONS.keys()), index=0)
    accent       = COLOR_OPTIONS[accent_label]

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    top_n          = st.slider("Recommendations", 5, 20, 10)
    use_clustering = st.toggle("Mood cluster filter", value=True)

    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    examples = [
    "Lofi night music",
    "Chill acoustic for studying",
    "Sad romantic songs",
    "Fast EDM party songs",
    "Classical piano for focus",
    "Happy pop for morning",
    ]
    for ex in examples:
        st.markdown(f"• *{ex}*")

# ── Dynamic theme variables ───────────────────────────────────
if dark_mode:
    BG       = "#0d0d0d"
    BG2      = "#1a1a1a"
    BG3      = "#222222"
    BORDER   = "#2a2a2a"
    TEXT1    = "#ffffff"
    TEXT2    = "#e0e0e0"
    TEXT3    = "#aaaaaa"
    INPUT_BG = "#1a1a1a"
    CARD_BG  = "#1a1a1a"
    EXP_BG   = "#0d1f0d"
else:
    BG       = "#f5f5f5"
    BG2      = "#ffffff"
    BG3      = "#eeeeee"
    BORDER   = "#dddddd"
    TEXT1    = "#111111"
    TEXT2    = "#444444"
    TEXT3    = "#888888"
    INPUT_BG = "#ffffff"
    CARD_BG  = "#ffffff"
    EXP_BG   = "#e8f5e9"

# accent with opacity
accent_light = accent + "22"
accent_mid   = accent + "55"

# ── Inject Precise CSS ────────────────────────────────────────
st.markdown(f"""
<style>
  /* Base Application Global Theme */
  .stApp {{ background-color: {BG}; color: {TEXT1}; }}
  
  /* Sidebar Container Structure */
  section[data-testid="stSidebar"] {{
      background-color: {BG2} !important;
      border-right: 1px solid {BORDER};
  }}
  /* This specific addition forces widget labels (Accent color, Dark mode, etc.) to remain visible */
  section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p {{
      color: {TEXT1} !important;
  }}
  section[data-testid="stSidebar"] .stMarkdown p {{
      color: {TEXT1} !important;
  }}
  section[data-testid="stSidebar"] h3 {{
      color: {TEXT1} !important;
  }}

  /* CRITICAL: Fix Selectbox White Invisible Text in Dark Mode & Fix Dropdown Options */
  div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
      background-color: {INPUT_BG} !important;
      color: {TEXT1} !important;
      border: 1px solid {BORDER} !important;
  }}
  div[data-testid="stSelectbox"] svg {{
      fill: {TEXT1} !important;
  }}
  /* Dropdown expanded popover visibility fix */
  ul[role="listbox"] {{
      background-color: {BG2} !important;
  }}
  ul[role="listbox"] li {{
      color: {TEXT1} !important;
      background-color: transparent !important;
  }}
  ul[role="listbox"] li:hover {{
      background-color: {accent_light} !important;
  }}

  /* CRITICAL: Safe Slider styling (Prevents huge solid block anomalies) */
  .stSlider div[data-baseweb="slider"] {{
      background-color: transparent !important;
  }}
  .stSlider [data-testid="stMetricValue"] {{
      color: {accent} !important;
  }}
  /* Track Highlight Color */
  .stSlider div[role="slider"] div[style*="left: 0%"] {{
      background: {accent} !important;
  }}

  /* CRITICAL: Safe Toggle Switch Background handling */
  div[data-testid="stCheckbox"] button[role="switch"][aria-checked="true"] {{
      background-color: {accent} !important;
  }}
  div[data-testid="stCheckbox"] button[role="switch"][aria-checked="false"] {{
      background-color: {TEXT3} !important;
  }}

  /* Main Input Elements */
  .stTextInput > div > div > input {{
      background-color: {INPUT_BG} !important;
      border: 2px solid {accent} !important;
      border-radius: 50px !important;
      color: {TEXT1} !important;
      font-size: 1rem !important;
      padding: 14px 22px !important;
  }}
  .stTextInput > div > div > input::placeholder {{ color: {TEXT3} !important; }}

  /* Buttons */
  .stButton > button {{
      background: {accent} !important;
      color: white !important; font-weight: 700 !important;
      border-radius: 50px !important; border: none !important;
      padding: 14px 32px !important; font-size: 1rem !important;
      width: 100% !important;
  }}
  .stButton > button:hover {{ opacity: 0.88 !important; }}

  /* Text Branding Structure */
  .main-title {{
      font-size: 2.6rem; font-weight: 800;
      background: linear-gradient(135deg, {accent}, {TEXT1});
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .sub-title {{ color: {TEXT2}; font-size:0.9rem; margin-top:2px; margin-bottom:20px; }}

  /* Metric Cards Layout */
  .metric-card {{
      background: {BG2}; border: 1px solid {BORDER};
      border-radius: 14px; padding: 16px;
      text-align: center;
  }}
  .metric-label {{ color:{TEXT2}; font-size:0.72rem; text-transform:uppercase; letter-spacing:1px; }}
  .metric-value {{ color:{accent}; font-size:1.5rem; font-weight:800; }}

  /* Recommendations Output Display */
  .track-card {{
      background: {CARD_BG}; border: 1px solid {BORDER};
      border-radius: 16px; padding: 20px 22px;
      margin-bottom: 14px;
  }}
  .track-card:hover {{ border-color: {accent}; }}
  .track-number {{ color:{accent}; font-size:1rem; font-weight:800; }}
  .track-name   {{ font-size:1.05rem; font-weight:700; color:{TEXT1}; }}
  .track-artist {{ color:{TEXT2}; font-size:0.85rem; margin-top:3px; }}
  .track-genre  {{
      background:{accent_light}; color:{accent};
      border: 1px solid {accent_mid};
      border-radius:20px; padding:3px 12px;
      font-size:0.72rem; font-weight:600;
      display:inline-block; margin-top:8px;
  }}
  .exp-box {{
      background:{EXP_BG}; border-left:3px solid {accent};
      border-radius:8px; padding:10px 14px;
      color:{TEXT2}; font-size:0.83rem; margin-top:12px;
  }}
  .score-badge {{
      background:{accent_light}; color:{accent};
      border-radius:8px; padding:4px 10px;
      font-size:0.78rem; font-weight:700;
  }}
  .link-btn {{
      display:inline-block; padding:7px 16px;
      border-radius:20px; font-size:0.8rem;
      font-weight:700; text-decoration:none;
      margin-right:8px; margin-top:12px;
      transition: opacity 0.2s;
  }}
  .link-btn:hover {{ opacity:0.8; }}
  .spotify-btn {{ background:#1DB954; color:#fff !important; }}
  .youtube-btn {{ background:#FF0000; color:#fff !important; }}
  hr {{ border-color:{BORDER} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Header Layout ─────────────────────────────────────────────
st.markdown(f'<div class="main-title">🎵 Music Recommender</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Agentic AI • KNN + Cosine Similarity + K-Means • '
            f'<span style="color:{accent}">■</span> {accent_label}</div>', unsafe_allow_html=True)

df, df_clustered, X_scaled, scaler, knn_model, kmeans = load_pipeline()

# ════════════════════════════════════════════════════════════════
# SEARCH
# ════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
query = st.text_input(
    "query", label_visibility="collapsed",
    placeholder="e.g. chill acoustic songs for late night studying..."
)
recommend_btn = st.button("🎧 Get Recommendations", type="primary")

# ════════════════════════════════════════════════════════════════
# RESULTS
# ════════════════════════════════════════════════════════════════
if recommend_btn:
    if not query.strip():
        st.warning("Please enter a query!")
        st.stop()

    with st.spinner("🤖 Analyzing your request..."):
        user_features = parse_intent(query)
        user_vec      = transform_user_vector(user_features, scaler)

        if use_clustering:
            cluster_id   = get_user_cluster(user_vec, kmeans)
            filtered_df  = filter_by_cluster(df_clustered, cluster_id)
            cluster_mask = df_clustered['cluster'] == cluster_id
            X_filtered   = X_scaled[cluster_mask.values]
            if len(filtered_df) < top_n * 2:
                filtered_df = df
                X_filtered  = X_scaled
            knn_f   = build_knn_model(X_filtered, n_neighbors=min(20, len(filtered_df)))
            results = get_recommendations(
                user_vec, X_filtered,
                filtered_df.reset_index(drop=True),
                knn_f, top_n=top_n
            )
            mood_label = filtered_df['mood'].iloc[0] if 'mood' in filtered_df.columns else "—"
        else:
            results    = get_recommendations(user_vec, X_scaled, df, knn_model, top_n=top_n)
            mood_label = "All moods"

        explanations = explain_all(results, user_features)

    # ── Intent metrics ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### 🧠 Agent Detected Intent")
    if use_clustering:
        st.markdown(f'Mood cluster: <span style="color:{accent}; font-weight:700; font-size:1.1rem;">{mood_label}</span>', unsafe_allow_html=True)

    cols = st.columns(5)
    metrics = [
        ("⚡ Energy",       user_features.get("energy", 0)),
        ("💃 Danceability", user_features.get("danceability", 0)),
        ("😊 Valence",      user_features.get("valence", 0)),
        ("🎸 Acousticness", user_features.get("acousticness", 0)),
        ("🥁 Tempo BPM",    user_features.get("tempo", 0)),
    ]
    for col, (label, val) in zip(cols, metrics):
        display = round(val, 1) if val > 10 else round(val, 2)
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{accent} !important;">{display}</div>
        </div>""", unsafe_allow_html=True)

    # ── Track cards ───────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### 🎶 Top {top_n} Recommendations")
    st.markdown("<br>", unsafe_allow_html=True)

    bar_fn = feat_bar if dark_mode else feat_bar_light

    for i, (_, row) in enumerate(results.iterrows()):
        t_name   = str(row['track_name'])
        t_artist = str(row['artists'])
        t_genre  = str(row['track_genre'])

        sp_url = spotify_url(t_name, t_artist)
        yt_url = youtube_url(t_name, t_artist)

        c1, c2 = st.columns([2, 3])

        with c1:
            st.markdown(f"""
            <div class="track-card">
              <div>
                <span class="track-number" style="color:{accent} !important;">#{i+1}</span>
                <span class="track-name">{t_name}</span>
              </div>
              <div class="track-artist">🎤 {t_artist}</div>
              <div class="track-genre" style="background:{accent_light}; color:{accent}; border-color:{accent_mid};">{t_genre}</div>
              <div style="margin-top:12px">
                <span class="score-badge" style="background:{accent_light}; color:{accent};">Match: {row['final_score']:.3f}</span>
                <span style="color:{TEXT3};font-size:0.78rem;margin-left:10px">
                  Popularity: {int(row.get('popularity', 0))}/100
                </span>
              </div>
              <div>
                <a href="{sp_url}" target="_blank" class="link-btn spotify-btn">
                  ▶ Play on Spotify
                </a>
                <a href="{yt_url}" target="_blank" class="link-btn youtube-btn">
                  ▶ Watch on YouTube
                </a>
              </div>
              <div class="exp-box" style="border-left-color:{accent};">🤖 {explanations[i]}</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            track_energy   = row.get('energy', user_features.get("energy", 0.5))
            track_dance    = row.get('danceability', user_features.get("danceability", 0.5))
            track_valence  = row.get('valence', user_features.get("valence", 0.5))
            track_acoustic = row.get('acousticness', user_features.get("acousticness", 0.5))
            track_tempo    = row.get('tempo', user_features.get("tempo", 120.0))

            st.markdown(f"""
            <div class="track-card">
              <div style="font-size:0.75rem;color:{TEXT3};
                          text-transform:uppercase;letter-spacing:1px;
                          margin-bottom:14px">Audio Features</div>
              {bar_fn("⚡ Energy",        track_energy,   accent)}
              {bar_fn("💃 Danceability",  track_dance,    accent)}
              {bar_fn("😊 Valence",       track_valence,  accent)}
              {bar_fn("🎸 Acousticness",  track_acoustic, accent)}
              {bar_fn("🥁 Tempo",         track_tempo,    accent, max_val=220)}
            </div>""", unsafe_allow_html=True)