import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


def build_knn_model(X_scaled: np.ndarray, n_neighbors: int = 20):
    """
    Fit a KNN model on the scaled feature matrix.
    n_neighbors=20 taaki baad mein cosine se re-rank kar sakein.
    """
    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric='euclidean',
        algorithm='ball_tree',
        n_jobs=-1
    )
    model.fit(X_scaled)
    print(f"[similarity] KNN model fitted on {X_scaled.shape[0]} tracks")
    return model


def get_recommendations(
    user_vector: np.ndarray,
    X_scaled: np.ndarray,
    df: pd.DataFrame,
    knn_model: NearestNeighbors,
    top_n: int = 10
) -> pd.DataFrame:
    """
    1. KNN se top 20 candidates nikalega
    2. Cosine similarity se re-rank karega
    3. Final top_n tracks return karega with scores
    """

    # ── Step 1: KNN candidates ───────────────────────────────
    distances, indices = knn_model.kneighbors(user_vector)
    candidate_indices = indices[0]          # shape: (20,)
    candidate_distances = distances[0]      # shape: (20,)

    # ── Step 2: Cosine similarity re-ranking ─────────────────
    candidate_vectors = X_scaled[candidate_indices]   # (20, 9)
    cos_scores = cosine_similarity(user_vector, candidate_vectors)[0]  # (20,)

    # ── Step 3: Build result dataframe ───────────────────────
    results = df.iloc[candidate_indices].copy()
    results['knn_distance'] = candidate_distances
    results['cosine_score'] = cos_scores

    # ── Step 4: Normalize knn distance → similarity score ────
    max_dist = results['knn_distance'].max() + 1e-9
    results['knn_score'] = 1 - (results['knn_distance'] / max_dist)

    # ── Step 5: Combined score (60% cosine + 40% knn) ────────
    results['final_score'] = (
        0.6 * results['cosine_score'] +
        0.4 * results['knn_score']
    )

    # ── Step 6: Sort and return top_n ────────────────────────
    results = results.sort_values('final_score', ascending=False)
    results = results.head(top_n)
    results.reset_index(drop=True, inplace=True)

    return results[['track_name', 'artists', 'track_genre',
                     'popularity', 'danceability', 'energy',
                     'valence', 'tempo', 'final_score']]


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from recommender.preprocessor import load_and_clean
    from recommender.feature_engine import build_feature_matrix, transform_user_vector

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, 'data', 'spotify_tracks.csv')

    # Load + process
    df = load_and_clean(path)
    X_scaled, scaler, features = build_feature_matrix(df)

    # Build KNN
    knn_model = build_knn_model(X_scaled, n_neighbors=20)

    # Dummy user query — "high energy workout song"
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
    results = get_recommendations(user_vec, X_scaled, df, knn_model, top_n=10)

    print("\n=== Top 10 Recommendations ===")
    print(results.to_string(index=False))