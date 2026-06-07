import re

# ── Keyword → Feature mapping ─────────────────────────────────────────────────
# Har keyword ek partial feature dict map karta hai

INTENT_MAP = {

    # ── Energy / Mood ────────────────────────────────────────
    "energetic":    {"energy": 0.90, "loudness": -4.0,  "valence": 0.75, "tempo": 135.0},
    "high energy":  {"energy": 0.92, "loudness": -3.0,  "valence": 0.80, "tempo": 140.0},
    "low energy":   {"energy": 0.25, "loudness": -14.0, "valence": 0.35, "tempo": 85.0},
    "workout":      {"energy": 0.93, "loudness": -3.5,  "danceability": 0.75, "tempo": 145.0},
    "gym":          {"energy": 0.95, "loudness": -3.0,  "tempo": 150.0},

    # ── Mood ─────────────────────────────────────────────────
    "happy":        {"valence": 0.85, "energy": 0.72, "danceability": 0.75},
    "sad":          {"valence": 0.15, "energy": 0.30, "acousticness": 0.70},
    "angry":        {"energy": 0.90, "valence": 0.20, "loudness": -3.0},
    "romantic":     {"valence": 0.65, "acousticness": 0.60, "energy": 0.40, "tempo": 90.0},
    "love":         {"valence": 0.70, "acousticness": 0.55, "energy": 0.38},
    "melancholic":  {"valence": 0.12, "energy": 0.28, "acousticness": 0.75},
    "chill":        {"energy": 0.35, "acousticness": 0.65, "tempo": 90.0,  "valence": 0.50},
    "relaxing":     {"energy": 0.30, "acousticness": 0.70, "tempo": 85.0},
    "calm":         {"energy": 0.28, "acousticness": 0.72, "loudness": -14.0},
    "aggressive":   {"energy": 0.92, "loudness": -2.5,  "valence": 0.18, "tempo": 155.0},
    "focused":      {"instrumentalness": 0.75, "energy": 0.50, "speechiness": 0.04},
    "study":        {"instrumentalness": 0.80, "energy": 0.40, "acousticness": 0.55},
    "focus":        {"instrumentalness": 0.78, "energy": 0.45, "speechiness": 0.03},

    # ── Danceability ─────────────────────────────────────────
    "dance":        {"danceability": 0.90, "energy": 0.82, "tempo": 125.0},
    "party":        {"danceability": 0.88, "energy": 0.85, "valence": 0.80, "tempo": 128.0},
    "groovy":       {"danceability": 0.85, "valence": 0.78, "tempo": 115.0},

    # ── Tempo ────────────────────────────────────────────────
    "fast":         {"tempo": 150.0, "energy": 0.80},
    "slow":         {"tempo": 75.0,  "energy": 0.30},
    "upbeat":       {"tempo": 130.0, "valence": 0.80, "danceability": 0.78},
    "slow tempo":   {"tempo": 70.0},
    "fast tempo":   {"tempo": 155.0},

    # ── Acoustic / Instrumental ──────────────────────────────
    "acoustic":     {"acousticness": 0.88, "energy": 0.35, "instrumentalness": 0.20},
    "instrumental": {"instrumentalness": 0.90, "speechiness": 0.03},
    "piano":        {"acousticness": 0.80, "instrumentalness": 0.70, "energy": 0.30},
    "guitar":       {"acousticness": 0.75, "energy": 0.45},

    # ── Genre hints ──────────────────────────────────────────
    "hip hop":      {"speechiness": 0.22, "danceability": 0.82, "energy": 0.68},
    "rap":          {"speechiness": 0.28, "danceability": 0.80, "energy": 0.72},
    "pop":          {"danceability": 0.72, "energy": 0.68, "valence": 0.65},
    "rock":         {"energy": 0.85, "loudness": -5.0,  "acousticness": 0.10},
    "metal":        {"energy": 0.95, "loudness": -2.0,  "valence": 0.20, "acousticness": 0.05},
    "jazz":         {"acousticness": 0.70, "instrumentalness": 0.40, "tempo": 95.0},
    "classical":    {"acousticness": 0.92, "instrumentalness": 0.88, "energy": 0.22},
    "edm":          {"danceability": 0.85, "energy": 0.90, "tempo": 132.0},
    "electronic":   {"danceability": 0.80, "energy": 0.82, "acousticness": 0.05},
    "bollywood":    {"danceability": 0.78, "valence": 0.72, "energy": 0.70},
    "indie":        {"acousticness": 0.55, "energy": 0.55, "valence": 0.55},
    "soul":         {"valence": 0.65, "acousticness": 0.60, "energy": 0.50},
    "blues":        {"acousticness": 0.65, "energy": 0.55, "valence": 0.40},

    # ── Time of day ──────────────────────────────────────────
    "morning":      {"valence": 0.72, "energy": 0.60, "acousticness": 0.40},
    "night":        {"energy": 0.45, "acousticness": 0.55, "valence": 0.40},
    "midnight":     {"energy": 0.35, "acousticness": 0.65, "valence": 0.30},
    "evening":      {"energy": 0.50, "valence": 0.55, "acousticness": 0.50},
    "sleep":        {"energy": 0.18, "acousticness": 0.85, "instrumentalness": 0.70, "tempo": 65.0},
}

# Default feature vector jab koi match na ho
DEFAULT_FEATURES = {
    "danceability":     0.60,
    "energy":           0.65,
    "loudness":         -8.0,
    "speechiness":      0.08,
    "acousticness":     0.35,
    "instrumentalness": 0.10,
    "liveness":         0.18,
    "valence":          0.55,
    "tempo":            115.0,
}


def parse_intent(user_text: str) -> dict:
    """
    Natural language text → audio feature dict.

    Example:
      Input:  "I want chill acoustic songs for studying at night"
      Output: {"energy": 0.34, "acousticness": 0.71, ...}
    """

    text = user_text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)   # punctuation remove

    # Start with defaults
    features = DEFAULT_FEATURES.copy()
    matched_keywords = []

    # Multi-word keywords pehle check karo (longer match priority)
    sorted_keywords = sorted(INTENT_MAP.keys(), key=len, reverse=True)

    for keyword in sorted_keywords:
        if keyword in text:
            matched_keywords.append(keyword)
            keyword_features = INTENT_MAP[keyword]
            # Average blending — multiple keywords smooth ho jayein
            for feat, val in keyword_features.items():
                if feat in features:
                    features[feat] = round((features[feat] + val) / 2, 4)
                else:
                    features[feat] = val

    if matched_keywords:
        print(f"[intent_parser] Matched keywords: {matched_keywords}")
    else:
        print(f"[intent_parser] No keywords matched — using defaults")

    print(f"[intent_parser] Extracted features: {features}")
    return features


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    test_queries = [
        "I want energetic workout songs",
        "Something chill and acoustic for studying at night",
        "Sad romantic bollywood songs",
        "Fast EDM party songs",
        "Classical piano instrumental for focus",
        "Happy upbeat pop songs for morning",
        "kuch bhi",                          # gibberish → defaults
    ]

    print("=" * 60)
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        result = parse_intent(q)
        print("-" * 40)