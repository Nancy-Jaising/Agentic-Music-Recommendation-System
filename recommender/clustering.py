import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Mood labels jo hum clusters ko assign karenge
MOOD_LABELS = {
    0: "Energetic / Party",
    1: "Chill / Relaxed",
    2: "Happy / Upbeat",
    3: "Sad / Melancholic",
    4: "Focus / Instrumental",
    5: "Aggressive / Intense",
    6: "Romantic / Soft",
    7: "Dance / Groovy"
}

N_CLUSTERS = len(MOOD_LABELS)   # 8 clusters


def build_clusters(X_scaled: np.ndarray, df: pd.DataFrame, random_state: int = 42):
    """
    Fit K-Means on scaled features.
    Returns:
      - df with 'cluster' and 'mood' columns added
      - fitted kmeans model
      - cluster centers
    """

    print(f"[clustering] Fitting K-Means with {N_CLUSTERS} clusters...")

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        init='k-means++',
        n_init=10,
        random_state=random_state
    )
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Cluster assign karo df mein
    df = df.copy()
    df['cluster'] = cluster_labels
    df['mood'] = df['cluster'].map(MOOD_LABELS)

    # Silhouette score — cluster quality check
    sample_size = min(10000, len(X_scaled))
    sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
    score = silhouette_score(X_scaled[sample_idx], cluster_labels[sample_idx])
    print(f"[clustering] Silhouette Score: {score:.4f}  (higher = better clusters)")

    # Cluster summary print karo
    print("\n[clustering] Cluster Distribution:")
    summary = df.groupby(['cluster', 'mood']).size().reset_index(name='count')
    print(summary.to_string(index=False))

    return df, kmeans, kmeans.cluster_centers_


def get_user_cluster(user_vector: np.ndarray, kmeans: KMeans) -> int:
    """
    User ke feature vector ka cluster predict karo.
    """
    cluster_id = kmeans.predict(user_vector)[0]
    mood = MOOD_LABELS.get(cluster_id, "Unknown")
    print(f"[clustering] User query matched → Cluster {cluster_id}: '{mood}'")
    return cluster_id


def filter_by_cluster(df: pd.DataFrame, cluster_id: int) -> pd.DataFrame:
    """
    Sirf usi cluster ke tracks return karo.
    """
    filtered = df[df['cluster'] == cluster_id].copy()
    print(f"[clustering] Tracks in cluster {cluster_id}: {len(filtered)}")
    return filtered


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from recommender.preprocessor import load_and_clean
    from recommender.feature_engine import build_feature_matrix, transform_user_vector

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, 'data', 'spotify_tracks.csv')

    df = load_and_clean(path)
    X_scaled, scaler, features = build_feature_matrix(df)

    # Clusters banao
    df_clustered, kmeans, centers = build_clusters(X_scaled, df)

    # User query test
    user_query = {
        'danceability': 0.8,
        'energy': 0.95,
        'loudness': -3.0,
        'speechiness': 0.08,
        'acousticness': 0.02,
        'instrumentalness': 0.0,
        'liveness': 0.15,
        'valence': 0.75,
        'tempo': 140.0
    }

    user_vec = transform_user_vector(user_query, scaler)
    cluster_id = get_user_cluster(user_vec, kmeans)
    filtered_df = filter_by_cluster(df_clustered, cluster_id)

    print(f"\nSample tracks from cluster:")
    print(filtered_df[['track_name', 'artists', 'mood', 'energy', 'danceability']].head(5))