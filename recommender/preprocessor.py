import pandas as pd
import os

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load raw Spotify CSV and return a cleaned DataFrame.
    """

    # ── 1. Load ──────────────────────────────────────────────
    df = pd.read_csv(filepath)
    print(f"[preprocessor] Loaded: {df.shape[0]} rows, {df.shape[1]} cols")

    # ── 2. Drop useless columns ──────────────────────────────
    drop_cols = ['Unnamed: 0', 'track_id', 'album_name', 'key',
                 'mode', 'time_signature', 'duration_ms']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    print(f"[preprocessor] After dropping cols: {df.shape}")

    # ── 3. Drop duplicates ───────────────────────────────────
    before = len(df)
    df.drop_duplicates(subset=['track_name', 'artists'], inplace=True)
    print(f"[preprocessor] Duplicates removed: {before - len(df)}")

    # ── 4. Drop rows with nulls ──────────────────────────────
    df.dropna(inplace=True)
    print(f"[preprocessor] After null drop: {df.shape[0]} rows")

    # ── 5. Convert explicit (bool → int) ────────────────────
    df['explicit'] = df['explicit'].astype(int)

    # ── 6. Reset index ───────────────────────────────────────
    df.reset_index(drop=True, inplace=True)

    print("[preprocessor] Cleaning done!\n")
    return df


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, 'data', 'spotify_tracks.csv')
    df = load_and_clean(path)
    print(df.head())
    print("\nFinal columns:", df.columns.tolist())
    print("Final shape:", df.shape)