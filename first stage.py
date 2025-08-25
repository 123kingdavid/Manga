import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# ============================
# Data: Expandable in-code catalog
# ============================
ITEMS = [
    # --- Manga ---
    {"title": "Naruto", "type": "Manga", "themes": "Ninja, friendship, perseverance, destiny, rivalry, training arcs, shonen", "synopsis": "A young ninja with a sealed beast seeks recognition and dreams of becoming Hokage."},
    {"title": "One Piece", "type": "Manga", "themes": "Pirates, adventure, friendship, treasure, freedom, found family, grand journey", "synopsis": "Luffy sets sail to find the One Piece and become Pirate King."},
    {"title": "Attack on Titan", "type": "Manga", "themes": "Humanity vs Titans, survival, war, tragedy, politics, plot twists, dark fantasy", "synopsis": "Humanity fights man-eating Titans while uncovering deep conspiracies about their world."},
    {"title": "Death Note", "type": "Manga", "themes": "Mystery, supernatural, justice, morality, mind games, cat-and-mouse", "synopsis": "A student finds a notebook that kills anyone whose name is written in it, spurring a duel with a genius detective."},
    {"title": "Bleach", "type": "Manga", "themes": "Shinigami, spirits, battles, friendship, supernatural, cool powers", "synopsis": "Ichigo becomes a Soul Reaper and protects both the living and the dead from hollows."},
    {"title": "Fullmetal Alchemist", "type": "Manga", "themes": "Alchemy, brotherhood, sacrifice, power, adventure, philosophy", "synopsis": "Two brothers use alchemy to restore their bodies after a forbidden ritual goes wrong."},
    {"title": "Demon Slayer", "type": "Manga", "themes": "Demons, family, swords, bonds, tragedy, breathing styles", "synopsis": "Tanjiro hunts demons to cure his sister and avenge his family."},
    {"title": "Jujutsu Kaisen", "type": "Manga", "themes": "Curses, sorcery, dark powers, friendship, battles, modern fantasy", "synopsis": "A student swallows a cursed object and joins jujutsu sorcerers to fight deadly curses."},
    {"title": "Chainsaw Man", "type": "Manga", "themes": "Devils, gore, dark comedy, survival, chaos, antihero", "synopsis": "Denji merges with his chainsaw devil and gets dragged into violent devil hunts."},
    {"title": "Black Clover", "type": "Manga", "themes": "Magic knights, rivalry, teamwork, adventure, destiny, underdog", "synopsis": "Magicless Asta aims to become Wizard King through grit and anti-magic."},
    {"title": "Tokyo Ghoul", "type": "Manga", "themes": "Dark fantasy, ghouls, identity, survival, tragedy, body horror", "synopsis": "A college student becomes half-ghoul and struggles between human and monster worlds."},
    {"title": "Blue Lock", "type": "Manga", "themes": "Soccer, competition, egoism, rivalry, sports psychology", "synopsis": "Strikers are locked in a brutal program to forge Japan's ultimate goal-scorer."},
    {"title": "Vinland Saga", "type": "Manga", "themes": "Vikings, revenge, war, slavery, adventure, character growth", "synopsis": "A young warrior seeks revenge and later searches for true freedom in a violent age."},
    {"title": "Haikyuu!!", "type": "Manga", "themes": "Volleyball, teamwork, rivalry, growth, sports, inspiration", "synopsis": "Shoyo Hinata strives to excel in volleyball despite his short stature."},
    {"title": "My Hero Academia", "type": "Manga", "themes": "Superheroes, academy, quirks, rivalry, justice, mentorship", "synopsis": "In a world of superpowers, a powerless boy inherits a mighty quirk to become a hero."},
    {"title": "Hunter x Hunter", "type": "Manga", "themes": "Adventure, Nen, exams, strategy, arcs, friendship, morality", "synopsis": "Gon seeks his father and faces morally complex adventures with friends."},
    {"title": "Berserk", "type": "Manga", "themes": "Dark fantasy, tragedy, demons, fate, trauma, epic battles", "synopsis": "Guts, a mercenary with a colossal sword, battles demonic forces and his fate."},
    {"title": "Spy x Family", "type": "Manga", "themes": "Spy comedy, family, wholesome, espionage, school life", "synopsis": "A spy, an assassin, and a telepath pretend to be a family for a mission."},
    {"title": "Oshi no Ko", "type": "Manga", "themes": "Idol industry, reincarnation, mystery, showbiz, revenge", "synopsis": "Twins in the entertainment world seek truth behind a star's death."},

    # --- Manhwa ---
    {"title": "Solo Leveling", "type": "Manhwa", "themes": "Hunters, dungeons, overpowered MC, monsters, action, grind", "synopsis": "Weak hunter Sung Jinwoo becomes overpowered through a mysterious system."},
    {"title": "Tower of God", "type": "Manhwa", "themes": "Towers, tests, betrayal, friendship, destiny, complex worldbuilding", "synopsis": "Bam climbs a mysterious tower of tests to find his friend and the truth."},
    {"title": "The Breaker", "type": "Manhwa", "themes": "Martial arts, revenge, training, school life, action, masters", "synopsis": "A bullied student learns secret martial arts from a powerful teacher."},
    {"title": "Noblesse", "type": "Manhwa", "themes": "Vampires, nobles, school, loyalty, hidden powers, action", "synopsis": "A noble awakens after 820 years and protects friends from secret organizations."},
    {"title": "Omniscient Reader", "type": "Manhwa", "themes": "Apocalypse, scenarios, meta, reader becomes key, regression vibes", "synopsis": "A novel reader survives apocalyptic scenarios that mirror his favorite web novel."},
    {"title": "True Beauty", "type": "Manhwa", "themes": "Romance, school life, identity, social media, comedy", "synopsis": "A girl becomes popular using makeup but grapples with authentic self-worth."},
    {"title": "Weak Hero", "type": "Manhwa", "themes": "Bullying, street fights, strategy, school gangs, grit", "synopsis": "A clever student dismantles school bullies with brains and brutal efficiency."},
    {"title": "The Beginning After the End", "type": "Manhwa", "themes": "Reincarnation, magic academy, swords & sorcery, growth, secrets", "synopsis": "A king reincarnates into a magic world, balancing power with family bonds."},
    {"title": "Lookism", "type": "Manhwa", "themes": "Body image, social status, bullying, dual bodies, urban life", "synopsis": "A student swaps between two bodies—one handsome, one not—navigating harsh society."},
    {"title": "Solo Leveling: Ragnarok", "type": "Manhwa", "themes": "Sequel, hunters, legacy, action, monsters", "synopsis": "Continuation of Solo Leveling saga with new threats and successors."},

    # --- Anime (original or notable anime-first) ---
    {"title": "Cowboy Bebop", "type": "Anime", "themes": "Bounty hunters, space western, jazz, melancholy, episodic", "synopsis": "A crew of misfits chases bounties in space while confronting their pasts."},
    {"title": "Neon Genesis Evangelion", "type": "Anime", "themes": "Mecha, psychological, trauma, apocalypse, symbolism", "synopsis": "Teen pilots battle existential threats while wrestling with identity and despair."},
    {"title": "Code Geass", "type": "Anime", "themes": "Rebellion, mind control, chessmaster, politics, mecha, masks", "synopsis": "Exiled prince gains a power to command and leads a masked rebellion."},
    {"title": "Your Name", "type": "Anime", "themes": "Body swap, romance, time, fate, comet, drama", "synopsis": "Two teens mysteriously swap bodies and connect across time and memory."},
    {"title": "Steins;Gate", "type": "Anime", "themes": "Time travel, otaku lab, consequences, thriller, butterfly effect", "synopsis": "Friends accidentally invent time messaging and battle timelines to save loved ones."},
    {"title": "Made in Abyss", "type": "Anime", "themes": "Adventure, dark fantasy, mystery, relics, descent, body horror", "synopsis": "A girl descends a deadly abyss to find her mother, facing horrific truths."},
    {"title": "Violet Evergarden", "type": "Anime", "themes": "Healing, letters, post-war, emotions, growth, drama", "synopsis": "An ex-soldier learns to understand emotions by writing letters for others."},
    {"title": "Mob Psycho 100", "type": "Anime", "themes": "Psychic powers, comedy, growth, self-control, mentorship", "synopsis": "A mild-mannered psychic tries to live normally while dealing with spirits and scams."},
    {"title": "Psycho-Pass", "type": "Anime", "themes": "Dystopia, crime, surveillance, justice, cyberpunk", "synopsis": "Cops enforce a system that quantifies a person's criminal tendencies."},
    {"title": "Samurai Champloo", "type": "Anime", "themes": "Road trip, Edo hip-hop, sword fights, found family, style", "synopsis": "Three travelers search for a samurai who smells of sunflowers across Edo-era Japan."},

    # --- More manga/manhwa/anime to broaden theme coverage ---
    {"title": "Dr. Stone", "type": "Manga", "themes": "Science, civilization rebuild, friendship, innovation, survival", "synopsis": "After humanity petrifies, a genius revives and rebuilds civilization with science."},
    {"title": "The Promised Neverland", "type": "Manga", "themes": "Orphans, escape, horror, strategy, found family", "synopsis": "Gifted orphans discover a gruesome secret and plot a daring escape."},
    {"title": "Fairy Tail", "type": "Manga", "themes": "Guilds, magic, friendship, quests, adventure", "synopsis": "Wizards in a guild take on jobs and protect each other as family."},
    {"title": "Toradora!", "type": "Manga", "themes": "Romance, school life, tsundere, comedy, growth", "synopsis": "Two students help each other with their crushes and find love unexpectedly."},
    {"title": "Kaguya-sama: Love is War", "type": "Manga", "themes": "Rom-com, mind games, pride, student council, antics", "synopsis": "Two geniuses wage psychological warfare to make the other confess first."},
    {"title": "A Silent Voice", "type": "Manga", "themes": "Bullying, redemption, disability, friendship, forgiveness", "synopsis": "A former bully seeks forgiveness from a deaf girl he tormented."},
    {"title": "Pluto", "type": "Manga", "themes": "Robots, mystery, morality, sci-fi, Urasawa, adaptation", "synopsis": "Detective Gesicht uncovers a conspiracy involving robots and human hatred."},
    {"title": "Kingdom", "type": "Manga", "themes": "Warring States, strategy, warfare, rise to greatness, brotherhood", "synopsis": "An orphan aims to become a great general during China's Warring States period."},
    {"title": "Record of Ragnarok", "type": "Manga", "themes": "Mythology, gods vs humans, tournament, hype battles", "synopsis": "Human champions duel gods to decide humanity's fate in a grand tournament."},
    {"title": "Mushoku Tensei", "type": "Anime", "themes": "Isekai, reincarnation, growth, magic, redemption", "synopsis": "A shut-in reincarnates in a fantasy world and vows to live without regrets."},
    {"title": "Re:Zero", "type": "Anime", "themes": "Isekai, time loop, trauma, choices, psychological", "synopsis": "A boy dies and returns by 'save points', enduring loops to protect friends."},
    {"title": "Parasyte", "type": "Manga", "themes": "Body horror, aliens, identity, coexistence, morality", "synopsis": "A teen's right hand is taken over by an alien parasite—together they survive."},
    {"title": "Baki", "type": "Manga", "themes": "Martial arts, tournaments, over-the-top fights, grit", "synopsis": "Baki faces outrageous fighters to surpass his father, the strongest creature."},
    {"title": "King's Avatar", "type": "Anime", "themes": "eSports, comeback, team building, strategy, slice of life", "synopsis": "A pro gamer starts over to reclaim his throne with a new team."},
    {"title": "Classroom of the Elite", "type": "Anime", "themes": "School, social manipulation, meritocracy, hidden genius", "synopsis": "Students in a cutthroat school system scheme for status and survival."},
]

# Convert to DataFrame
DF = pd.DataFrame(ITEMS)

# ============================
# Text processing & model
# ============================
@st.cache_data(show_spinner=False)
def get_dataframe():
    return DF.copy()

@st.cache_resource(show_spinner=False)
def build_vectorizer_and_matrix(df: pd.DataFrame):
    # Combine fields to capture themes & storylines
    corpus = (df["themes"].fillna("") + " . " + df["synopsis"].fillna("") + " . " + df["type"].fillna("")).tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix

# Utility: Title matcher with tolerance
def find_title_idx(title: str, df: pd.DataFrame):
    titles = df["title"].tolist()
    if title in titles:
        return titles.index(title)
    # fuzzy match
    match = get_close_matches(title, titles, n=1, cutoff=0.6)
    if match:
        return titles.index(match[0])
    return None

# Core: recommend by title index or by free-text query
def recommend_similar(df: pd.DataFrame, vectorizer: TfidfVectorizer, matrix, query_vector, top_n=10, type_filter=None, exclude_idx=None):
    sims = cosine_similarity(query_vector, matrix)[0]
    # Build results
    indices = np.argsort(-sims)
    results = []
    for idx in indices:
        if exclude_idx is not None and idx == exclude_idx:
            continue
        row = df.iloc[idx]
        if type_filter and row["type"] not in type_filter:
            continue
        score = float(sims[idx])
        if score <= 0:
            continue
        results.append({
            "title": row["title"],
            "type": row["type"],
            "similarity": round(score, 4),
            "themes": row["themes"],
            "synopsis": row["synopsis"],
        })
        if len(results) >= top_n:
            break
    return results

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="AI Manga/Anime/Manhwa Recommender", page_icon="📚", layout="wide")

st.title("📚 AI Manga · Anime · Manhwa Recommender")
st.write("Find recommendations based on **themes** and **storylines** you love.")

with st.expander("About the model / How it works", expanded=False):
    st.markdown(
        """
        This app uses **TF‑IDF** text vectors over each title's *themes + synopsis + type*,
        then computes **cosine similarity** to surface the closest matches. You can:
        - Pick a known title (fuzzy search supported), or
        - Describe what you like in pure text (e.g., *"revenge, politics, mind games"*),
        then filter by Manga / Anime / Manhwa.
        """
    )

# Load data and model
_df = get_dataframe()
vectorizer, matrix = build_vectorizer_and_matrix(_df)

# Sidebar controls
st.sidebar.header("Filters")
content_types = st.sidebar.multiselect(
    "Include types",
    options=["Manga", "Anime", "Manhwa"],
    default=["Manga", "Anime", "Manhwa"],
)

k = st.sidebar.slider("How many results?", min_value=3, max_value=20, value=10, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: You can search by **title** or by **themes/storyline** text.")

# Tabs for modes
tab1, tab2 = st.tabs(["🔎 Recommend by Title", "📝 Recommend by Themes/Storyline Text"])

with tab1:
    st.subheader("Recommend by Title")
    title_input = st.text_input("Type a title you've read/watched", placeholder="e.g., Naruto, Solo Leveling, Code Geass")
    colA, colB = st.columns([1,1])
    with colA:
        go1 = st.button("Get Recommendations", type="primary")
    with colB:
        clear1 = st.button("Clear")
    if clear1:
        st.experimental_rerun()

    if go1:
        if not title_input.strip():
            st.warning("Please enter a title.")
        else:
            idx = find_title_idx(title_input.strip(), _df)
            if idx is None:
                st.error("Title not found. Try checking spelling or use the Themes/Storyline tab.")
                # show closest suggestions
                suggestions = get_close_matches(title_input.strip(), _df["title"].tolist(), n=5, cutoff=0.4)
                if suggestions:
                    st.info("Did you mean: " + ", ".join(suggestions))
            else:
                query_vec = matrix[idx]
                recs = recommend_similar(_df, vectorizer, matrix, query_vec, top_n=k, type_filter=content_types, exclude_idx=idx)
                st.success(f"Recommendations because you like **{_df.iloc[idx]['title']}**:")
                for r in recs:
                    with st.container(border=True):
                        st.markdown(f"### {r['title']} · *{r['type']}*  ")
                        st.markdown(f"**Similarity:** {r['similarity']}  ")
                        st.markdown(f"**Themes:** {r['themes']}")
                        st.markdown(f"**Synopsis:** {r['synopsis']}")

with tab2:
    st.subheader("Recommend by Themes / Storyline Text")
    prompt = st.text_area(
        "Describe what you like",
        placeholder="e.g., dark fantasy with mind games and political intrigue; revenge and tragedy; epic swords and demons",
        height=120,
    )
    col1, col2 = st.columns([1,1])
    with col1:
        go2 = st.button("Find Matches", type="primary", key="go2")
    with col2:
        clear2 = st.button("Clear Text", key="clear2")
    if clear2:
        st.experimental_rerun()

    if go2:
        text = (prompt or "").strip()
        if not text:
            st.warning("Please write some themes or storyline keywords.")
        else:
            query_vec = vectorizer.transform([text])
            recs = recommend_similar(_df, vectorizer, matrix, query_vec, top_n=k, type_filter=content_types)
            if not recs:
                st.error("No close matches. Try different or broader keywords.")
            else:
                st.success("Top matches for your description:")
                for r in recs:
                    with st.container(border=True):
                        st.markdown(f"### {r['title']} · *{r['type']}*  ")
                        st.markdown(f"**Similarity:** {r['similarity']}  ")
                        st.markdown(f"**Themes:** {r['themes']}")
                        st.markdown(f"**Synopsis:** {r['synopsis']}")

# Footer
st.markdown("---")
st.caption("Built with TF‑IDF cosine similarity. Add more titles in the ITEMS list to improve coverage.")
