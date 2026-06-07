# agent/explainer.py

def explain_recommendation(track: dict, user_features: dict) -> str:
    """
    Ek track aur user intent features leke
    human-readable explanation return karta hai.

    track      : dict (ek row from results DataFrame)
    user_features : dict (intent_parser ka output)
    """

    reasons = []

    # ── Energy ───────────────────────────────────────────────
    energy = track.get("energy", 0)
    u_energy = user_features.get("energy", 0.65)
    if u_energy >= 0.75 and energy >= 0.75:
        reasons.append(f"high energy ({energy:.2f}) matches your energetic vibe")
    elif u_energy <= 0.40 and energy <= 0.45:
        reasons.append(f"low energy ({energy:.2f}) suits your chill/relaxed mood")

    # ── Danceability ─────────────────────────────────────────
    dance = track.get("danceability", 0)
    u_dance = user_features.get("danceability", 0.60)
    if u_dance >= 0.75 and dance >= 0.75:
        reasons.append(f"highly danceable ({dance:.2f}) — great for dancing/party")
    elif u_dance <= 0.45 and dance <= 0.50:
        reasons.append(f"low danceability ({dance:.2f}) — calm and non-intrusive")

    # ── Valence (happiness) ───────────────────────────────────
    valence = track.get("valence", 0)
    u_valence = user_features.get("valence", 0.55)
    if u_valence >= 0.70 and valence >= 0.65:
        reasons.append(f"positive/happy tone (valence {valence:.2f})")
    elif u_valence <= 0.30 and valence <= 0.35:
        reasons.append(f"melancholic tone (valence {valence:.2f}) matches your sad mood")

    # ── Acousticness ─────────────────────────────────────────
    acoustic = track.get("acousticness", 0)
    u_acoustic = user_features.get("acousticness", 0.35)
    if u_acoustic >= 0.60 and acoustic >= 0.55:
        reasons.append(f"acoustic feel ({acoustic:.2f}) — natural and soothing")
    elif u_acoustic <= 0.15 and acoustic <= 0.15:
        reasons.append(f"electronic/produced sound ({acoustic:.2f})")

    # ── Instrumentalness ─────────────────────────────────────
    instrumental = track.get("instrumentalness", 0)
    u_instr = user_features.get("instrumentalness", 0.10)
    if u_instr >= 0.50 and instrumental >= 0.40:
        reasons.append(f"mostly instrumental ({instrumental:.2f}) — good for focus/study")

    # ── Tempo ─────────────────────────────────────────────────
    tempo = track.get("tempo", 0)
    u_tempo = user_features.get("tempo", 115.0)
    if u_tempo >= 130 and tempo >= 125:
        reasons.append(f"fast tempo ({tempo:.0f} BPM) — keeps energy high")
    elif u_tempo <= 90 and tempo <= 95:
        reasons.append(f"slow tempo ({tempo:.0f} BPM) — relaxed and easy-going")

    # ── Popularity bonus ─────────────────────────────────────
    popularity = track.get("popularity", 0)
    if popularity >= 70:
        reasons.append(f"very popular track (score: {popularity})")
    elif popularity <= 20:
        reasons.append(f"hidden gem (popularity: {popularity})")

    # ── Final explanation string ──────────────────────────────
    if reasons:
        explanation = "Recommended because: " + ", ".join(reasons) + "."
    else:
        explanation = "Recommended based on overall audio feature similarity to your request."

    return explanation


def explain_all(results_df, user_features: dict) -> list:
    """
    Poore results DataFrame ke liye explanations generate karo.
    Returns list of explanation strings.
    """
    explanations = []
    for _, row in results_df.iterrows():
        track = row.to_dict()
        exp = explain_recommendation(track, user_features)
        explanations.append(exp)
    return explanations


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from recommender.preprocessor import load_and_clean
    from recommender.feature_engine import build_feature_matrix, transform_user_vector
    from recommender.similarity import build_knn_model, get_recommendations
    from agent.intent_parser import parse_intent

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, 'data', 'spotify_tracks.csv')

    # Pipeline
    df         = load_and_clean(path)
    X_scaled, scaler, _ = build_feature_matrix(df)
    knn_model  = build_knn_model(X_scaled, n_neighbors=20)

    # User query
    query         = "I want chill acoustic songs for studying at night"
    user_features = parse_intent(query)
    user_vec      = transform_user_vector(user_features, scaler)

    # Recommendations
    results = get_recommendations(user_vec, X_scaled, df, knn_model, top_n=5)

    # Explanations
    explanations = explain_all(results, user_features)

    print("\n=== Recommendations + Explanations ===\n")
    for i, (_, row) in enumerate(results.iterrows()):
        print(f"{i+1}. {row['track_name']} — {row['artists']}")
        print(f"   Genre: {row['track_genre']} | Popularity: {row['popularity']}")
        print(f"   {explanations[i]}")
        print()