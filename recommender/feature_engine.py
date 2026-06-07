import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Features jo recommendation mein use honge ────────────────
AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

def build_feature_matrix(df: pd.DataFrame):
    """
    Scale audio features and return:
      - scaled numpy array  (for KNN / cosine)
      - fitted scaler       (for transforming user query later)
      - feature column list
    """

    # ── 1. Select only audio feature columns ─────────────────
    X = df[AUDIO_FEATURES].copy()

    # ── 2. Fit StandardScaler ─────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"[feature_engine] Feature matrix shape: {X_scaled.shape}")
    print(f"[feature_engine] Features used: {AUDIO_FEATURES}")

    return X_scaled, scaler, AUDIO_FEATURES

def transform_user_vector(user_features: dict, scaler: StandardScaler) -> np.ndarray:
    """
    Convert a user intent dict into a scaled feature vector.
    """
    # Dict → DataFrame (fixes sklearn warning)
    vector = {f: [user_features.get(f, 0.0)] for f in AUDIO_FEATURES}
    vector_df = pd.DataFrame(vector)
    vector_scaled = scaler.transform(vector_df)
    return vector_scaled

# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from recommender.preprocessor import load_and_clean

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, 'data', 'spotify_tracks.csv')

    df = load_and_clean(path)
    X_scaled, scaler, features = build_feature_matrix(df)

    print("\nFirst row (scaled):", X_scaled[0])
    print("Scaler means:", np.round(scaler.mean_, 3))