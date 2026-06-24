# 🎵 Agentic Music Recommendation System
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agentic-music-recommendation-system-rwiw8krkyzshtharcgz9ov.streamlit.app/)
> 🚀 **Live Demo:** Try out the web application instantly without downloading any code here: [Agentic Music Recommender App](https://agentic-music-recommendation-system-rwiw8krkyzshtharcgz9ov.streamlit.app/)

---

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-Intent%20Parsing-green?style=for-the-badge)

---

## 📌 Overview

An **Agentic Music Recommendation System** that understands natural language mood/intent and recommends Spotify tracks using audio feature matching and KNN similarity — with explainable recommendations.

> "Give me chill acoustic songs for studying at night" → instantly get matched tracks with reasons why.

---

## ✨ Features

- 🧠 Natural language intent parsing (mood → audio features)
- 🎯 KNN-based similarity matching on Spotify audio features
- 💬 Explainable recommendations — tells you *why* each song was picked
- 🎨 Interactive Streamlit web app
- 📊 EDA notebook included

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| ML | Scikit-learn (KNN, StandardScaler) |
| NLP | Custom Intent Parser (keyword → feature mapping) |
| Web App | Streamlit |
| Data | Spotify Tracks Dataset (CSV) |
| Clustering | K-Means |

---

## 🧠 How It Works

1. User types a natural language query ("energetic gym music")
2. **Intent Parser** maps keywords → Spotify audio features (energy, tempo, valence, etc.)
3. **KNN model** finds most similar tracks in feature space
4. **Explainer** generates human-readable reasons for each recommendation

---

## 📂 Project Structure

```
Agentic-Music-Recommendation-System/
│
├── agent/
│   ├── intent_parser.py       # NLP: text → audio features
│   └── explainer.py           # Explainability layer
│
├── recommender/
│   ├── preprocessor.py        # Data cleaning
│   ├── feature_engine.py      # Feature matrix builder
│   ├── clustering.py          # K-Means clustering
│   └── similarity.py          # KNN similarity search
│
├── data/
│   └── spotify_tracks.csv     # Dataset
│
├── notebooks/
│   └── eda.ipynb              # Exploratory Data Analysis
│
├── app.py                     # Streamlit app entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Nancy-Jaising/Agentic-Music-Recommendation-System.git
cd Agentic-Music-Recommendation-System
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

👉 Open in browser: http://localhost:8501

---

## 📄 License

MIT License
