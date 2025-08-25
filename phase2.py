import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
def load_data():
    return pd.read_csv("dataset.csv")

# Recommend function
def recommend(df, user_input, filter_type=None):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['theme'].fillna(""))

    all_data = df['theme'].tolist() + [user_input]
    tfidf_all = vectorizer.fit_transform(all_data)

    cosine_sim = cosine_similarity(tfidf_all[-1], tfidf_all[:-1])
    scores = list(enumerate(cosine_sim[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in scores[:5]:
        rec = df.iloc[idx]
        if filter_type is None or rec['type'].lower() == filter_type.lower():
            results.append({
                'title': rec['title'],
                'type': rec['type'],
                'theme': rec['theme'],
                'image': rec['image'] if 'image' in df.columns else None
            })
    return results


# Streamlit config
st.set_page_config(page_title="AI Manga/Anime/Manhwa Recommender", page_icon="📚", layout="wide")

# 🌌 Custom CSS with anime wallpaper
st.markdown("""
    <style>
        .stApp {
            background: "yuji-itadori-megumi-3840x2160-20100"
            background-size: cover;
            background-attachment: fixed;
            color: #fff;
            font-family: 'Trebuchet MS', sans-serif;
        }
        h1, h2, h3 {
            color: #FFD700;
            text-align: center;
            text-shadow: 2px 2px 5px #000;
        }
        .stButton button {
            background-color: #ff4757;
            color: white;
            border-radius: 12px;
            padding: 0.6em 1em;
            font-size: 16px;
            font-weight: bold;
            border: none;
        }
        .stButton button:hover {
            background-color: #e84118;
        }
        .recommend-card {
            background: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0px 0px 15px rgba(255, 215, 0, 0.5);
        }
        .recommend-card img {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📚 AI Manga/Anime/Manhwa Recommender")
st.write("✨ Discover your next favorite story based on what you already love!")

df = load_data()

user_input = st.text_input("🔍 Enter a title or theme:")
filter_type = st.selectbox("📌 Filter by type:", ["All", "Manga", "Anime", "Manhwa"])

if st.button("🚀 Recommend"):
    if user_input.strip() == "":
        st.warning("Please enter a title or theme.")
    else:
        f_type = None if filter_type == "All" else filter_type
        recommendations = recommend(df, user_input, f_type)

        if recommendations:
            st.subheader("🔥 Recommended for you:")
            for rec in recommendations:
                with st.container():
                    st.markdown(f"<div class='recommend-card'>", unsafe_allow_html=True)
                    st.subheader(rec['title'])
                    st.write(f"**Type:** {rec['type']}")
                    st.write(f"**Theme:** {rec['theme']}")
                    if rec['image'] and pd.notna(rec['image']):
                        st.image(rec['image'], width=200)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("😢 No recommendations found. Try another theme or title.")
