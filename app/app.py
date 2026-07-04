import streamlit as st
import pandas as pd
import numpy as np
import re
import random
import os 
from collections import Counter

import plotly.express as px
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.xquik_export import normalize_xquik_export_columns

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Social Media Sentiment Analysis(Intelligence)",
    page_icon="📊",
    layout="wide"
)

css_path = os.path.join(os.path.dirname(__file__), "style.css")

with open(css_path, "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# =====================================================
# PREMIUM CSS
# =====================================================


# =====================================================
# TEXT CLEANING
# =====================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =====================================================
# ML TRAINING DATA
# =====================================================
@st.cache_resource
def train_sentiment_model():
    training_data = pd.DataFrame({
        "text": [
            "I love this app it is amazing and smooth",
            "The service was excellent and very fast",
            "Great product and very helpful support",
            "I am very happy with the experience",
            "This platform is awesome and easy to use",
            "The delivery was quick and perfect",
            "Customer support was polite and helpful",
            "The offer is great and I loved it",
            "Very satisfied with the product quality",
            "The app design is beautiful and simple",

            "Worst service ever I hate this app",
            "The delivery was very late and bad",
            "Payment failed and money was deducted",
            "Customer support did not reply",
            "The app keeps crashing again and again",
            "Very poor experience and terrible service",
            "The product was damaged and bad",
            "Refund process is slow and frustrating",
            "This is expensive and not worth it",
            "I am disappointed with the quality",

            "The service is okay",
            "It is an average experience",
            "Nothing special about this product",
            "The app is normal",
            "The delivery was neither fast nor slow",
            "I have no strong opinion",
            "The product is fine",
            "The update is acceptable",
            "It works as expected",
            "The experience is neutral"
        ],
        "sentiment": [
            "Positive","Positive","Positive","Positive","Positive",
            "Positive","Positive","Positive","Positive","Positive",
            "Negative","Negative","Negative","Negative","Negative",
            "Negative","Negative","Negative","Negative","Negative",
            "Neutral","Neutral","Neutral","Neutral","Neutral",
            "Neutral","Neutral","Neutral","Neutral","Neutral"
        ]
    })

    training_data["cleaned"] = training_data["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        training_data["cleaned"],
        training_data["sentiment"],
        test_size=0.25,
        random_state=42,
        stratify=training_data["sentiment"]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42)
    }

    results = []
    trained_models = {}

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "F1 Score": round(f1_score(y_test, y_pred, average="weighted") * 100, 2)
        })

        trained_models[name] = pipeline

    results_df = pd.DataFrame(results)
    best_model_name = results_df.sort_values(
        by=["F1 Score", "Accuracy"],
        ascending=False
    ).iloc[0]["Model"]

    best_model = trained_models[best_model_name]
    best_pred = best_model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        best_pred,
        labels=["Positive", "Negative", "Neutral"]
    )

    return best_model, results_df, cm, ["Positive", "Negative", "Neutral"], best_model_name


model, model_results_df, confusion_matrix_values, confusion_labels, best_model_name = train_sentiment_model()


# =====================================================
# BUSINESS LOGIC
# =====================================================
def predict_sentiment(text):
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]
    confidence = round(np.max(probabilities) * 100, 2)
    return prediction, confidence, cleaned


def detect_category(text):
    text = clean_text(text)

    categories = {
        "Delivery Issue": ["delivery", "late", "delay", "order", "rider", "shipping"],
        "Payment Issue": ["payment", "refund", "money", "transaction", "upi", "deducted"],
        "App Issue": ["app", "crash", "bug", "login", "loading", "screen", "update"],
        "Customer Support": ["support", "reply", "help", "customer", "agent", "complaint"],
        "Pricing Issue": ["price", "cost", "expensive", "charge", "fee"],
        "Product Quality": ["quality", "food", "product", "damaged", "taste", "packaging"],
        "Campaign Feedback": ["offer", "discount", "campaign", "ad", "promotion", "sale"]
    }

    matched = []
    for category, words in categories.items():
        if any(word in text for word in words):
            matched.append(category)

    return matched[0] if matched else "General Feedback"


def get_priority(sentiment, confidence):
    if sentiment == "Negative" and confidence >= 65:
        return "High"
    elif sentiment == "Negative":
        return "Medium"
    elif sentiment == "Neutral":
        return "Low"
    return "Positive"


def get_action(sentiment, category):
    if sentiment == "Negative":
        return f"Review urgently. Main issue detected: {category}."
    elif sentiment == "Positive":
        return "Use this feedback as testimonial or campaign proof."
    return "Monitor this feedback for future trend analysis."


def add_missing_columns(df):
    if "platform" not in df.columns:
        platforms = ["Instagram", "YouTube", "Twitter/X", "LinkedIn", "Facebook"]
        df["platform"] = [platforms[i % len(platforms)] for i in range(len(df))]

    if "campaign" not in df.columns:
        campaigns = ["Launch Campaign", "Festive Offer", "App Update", "Brand Awareness"]
        df["campaign"] = [campaigns[i % len(campaigns)] for i in range(len(df))]

    if "date" not in df.columns:
        dates = pd.date_range(end=pd.Timestamp.today(), periods=len(df)).date
        df["date"] = dates

    return df


def analyze_dataframe(df):
    df = add_missing_columns(df.copy())

    results = df["text"].apply(predict_sentiment)

    df["Sentiment"] = results.apply(lambda x: x[0])
    df["Confidence"] = results.apply(lambda x: x[1])
    df["Cleaned_Text"] = results.apply(lambda x: x[2])
    df["Category"] = df["text"].apply(detect_category)
    df["Priority"] = df.apply(lambda row: get_priority(row["Sentiment"], row["Confidence"]), axis=1)
    df["Recommended_Action"] = df.apply(lambda row: get_action(row["Sentiment"], row["Category"]), axis=1)

    return df


def create_sample_data():
    comments = [
        "I love this app, it is smooth and fast",
        "Worst delivery experience ever, my order was late",
        "The product quality is okay, nothing special",
        "Payment failed but money was deducted",
        "Customer support was very helpful and polite",
        "The app keeps crashing after login",
        "Amazing discount campaign, loved the offer",
        "Food quality was poor and delivery was delayed",
        "The service is average",
        "Great product and very fast delivery",
        "Refund process is too slow and frustrating",
        "The new update is excellent",
        "Price is too expensive compared to competitors",
        "Support team did not reply to my complaint",
        "I am happy with the overall experience",
        "The app UI is beautiful and easy to use",
        "Bad packaging and damaged product received",
        "Offer was good but delivery was late",
        "Normal service, nothing very good or bad",
        "I will recommend this platform to my friends",
        "The login screen has a bug",
        "The festive offer was amazing",
        "Product packaging was damaged",
        "The customer agent helped me quickly",
        "The app update is acceptable"
    ]

    return pd.DataFrame({"text": comments})

def normalize_real_dataset(df):
    df = normalize_xquik_export_columns(df)

    possible_text_columns = [
        "text", "tweet", "Tweet", "tweet_text", "full_text", "comment", "comments", "Comment",
        "content", "review", "Review", "selected_text",
        "message", "feedback", "Feedback", "body", "post"
    ]

    possible_sentiment_columns = [
        "sentiment", "Sentiment", "label", "Label", "category",
        "airline_sentiment", "sentiments", "target", "class"
    ]

    text_col = None
    sentiment_col = None

    for col in possible_text_columns:
        if col in df.columns:
            text_col = col
            break

    for col in possible_sentiment_columns:
        if col in df.columns:
            sentiment_col = col
            break

    if text_col is None:
        return None, "No valid text/comment column found."

    df = df.rename(columns={text_col: "text"})

    if sentiment_col is not None:
        df = df.rename(columns={sentiment_col: "Real_Sentiment"})
        df["Real_Sentiment"] = df["Real_Sentiment"].astype(str).str.strip().str.lower()

        mapping = {
            "positive": "Positive",
            "pos": "Positive",
            "1": "Positive",
            "4": "Positive",

            "negative": "Negative",
            "neg": "Negative",
            "-1": "Negative",
            "0": "Negative",

            "neutral": "Neutral",
            "neu": "Neutral",
            "2": "Neutral",

            "irrelevant": "Neutral",
            "mixed": "Neutral"
        }

        df["Real_Sentiment"] = df["Real_Sentiment"].map(mapping).fillna(df["Real_Sentiment"])

    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]

    return df, None   


def brand_health(df):
    total = len(df)
    if total == 0:
        return 0
    positive = (df["Sentiment"] == "Positive").sum()
    negative = (df["Sentiment"] == "Negative").sum()
    return round(((positive - negative) / total) * 100, 2)


def top_keywords(df, n=15):
    words = " ".join(df["Cleaned_Text"].astype(str)).split()
    words = [word for word in words if len(word) > 3]
    return pd.DataFrame(Counter(words).most_common(n), columns=["Keyword", "Frequency"])


# =====================================================
# COLOR PALETTES
# =====================================================
sentiment_colors = {
    "Positive": "#22C55E",
    "Negative": "#EF4444",
    "Neutral": "#3B82F6"
}

multi_colors = [
    "#22C55E", "#EF4444", "#3B82F6", "#F59E0B", "#A855F7",
    "#06B6D4", "#EC4899", "#84CC16", "#F97316", "#14B8A6"
]


# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <h1>📊 Social Media Sentiment Intelligence Dashboard</h1>
    <p>Advanced ML-powered social listening website for comments, reviews, complaints, campaigns, customer emotions, and brand reputation analytics.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="quote-box">
💬 “Behind every comment is a customer emotion. This dashboard converts public opinion into business intelligence.”
</div>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🚀 Sentiment Intelligence")
st.sidebar.markdown("Analyze comments, detect sentiment, monitor complaints, and generate business insights.")

page = st.sidebar.radio(
    "Choose Section",
    [
        "🏠 Executive Home",
        "✍️ Single Comment Analyzer",
        "📂 Bulk CSV Analyzer",
        "📈 Premium Dashboard",
        "🔍 Keyword Intelligence",
        "🚨 Complaint Monitor",
        "📊 Campaign Analytics",
        "🧠 ML Model Info",
        "📤 Download Report"
    ]
)

st.sidebar.markdown("---")

if st.sidebar.button("🧪 Load Professional Sample Data"):
    sample_df = create_sample_data()
    st.session_state["data"] = analyze_dataframe(sample_df)
    st.sidebar.success("Sample data loaded successfully!")


# =====================================================
# EXECUTIVE HOME
# =====================================================
if page == "🏠 Executive Home":
    st.markdown('<div class="section-title">🏠 Executive Project Overview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="glass-card">
            <h3>💬 Social Listening</h3>
            <p>Analyze customer comments, social media reviews, tweets, and feedback at scale.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card">
            <h3>🧠 ML Sentiment Engine</h3>
            <p>Uses TF-IDF and Logistic Regression to classify text as Positive, Negative, or Neutral.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="glass-card">
            <h3>📊 Business Intelligence</h3>
            <p>Generates brand health, complaint priority, keyword trends, and campaign insights.</p>
        </div>
        """, unsafe_allow_html=True)

    features = pd.DataFrame({
        "Premium Feature": [
            "ML Sentiment Classification",
            "Confidence Score",
            "Brand Health Score",
            "Complaint Category Detection",
            "Priority Tagging",
            "Platform Comparison",
            "Campaign Analytics",
            "Keyword Intelligence",
            "Business Action Recommendation",
            "Downloadable Report"
        ],
        "Recruiter Value": [
            "Shows NLP + ML knowledge",
            "Shows model interpretation",
            "Shows business KPI thinking",
            "Shows customer analytics logic",
            "Shows decision-making automation",
            "Shows dashboard analytics",
            "Shows marketing analytics use case",
            "Shows text mining ability",
            "Shows business intelligence skill",
            "Shows end-to-end project completion"
        ]
    })

    st.markdown("### ⭐ What Makes This Project Recruiter-Friendly?")
    st.dataframe(features, use_container_width=True)


# =====================================================
# SINGLE COMMENT ANALYZER
# =====================================================
elif page == "✍️ Single Comment Analyzer":
    st.markdown('<div class="section-title">✍️ Single Comment Sentiment Analyzer</div>', unsafe_allow_html=True)

    text = st.text_area(
        "Enter a customer comment, tweet, review, or feedback:",
        height=160,
        placeholder="Example: The app is amazing, but the delivery was very late."
    )

    if st.button("🚀 Analyze Sentiment"):
        if text.strip() == "":
            st.warning("Please enter a comment.")
        else:
            sentiment, confidence, cleaned = predict_sentiment(text)
            category = detect_category(text)
            priority = get_priority(sentiment, confidence)
            action = get_action(sentiment, category)

            card_color = "green" if sentiment == "Positive" else "red" if sentiment == "Negative" else "blue"

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(f"""
                <div class="metric-card {card_color}">
                    <div class="metric-value">{sentiment}</div>
                    <div class="metric-label">Predicted Sentiment</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card purple">
                    <div class="metric-value">{confidence}%</div>
                    <div class="metric-label">ML Confidence</div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="metric-card yellow">
                    <div class="metric-value">{priority}</div>
                    <div class="metric-label">Priority Level</div>
                </div>
                """, unsafe_allow_html=True)

            with c4:
                st.markdown(f"""
                <div class="metric-card cyan">
                    <div class="metric-value">TF-IDF</div>
                    <div class="metric-label">Feature Method</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 🧹 Cleaned Text")
            st.code(cleaned)

            st.markdown("### 🏷️ Business Category")
            st.success(category)

            st.markdown("### 💡 Recommended Business Action")
            st.info(action)


# =====================================================
# BULK CSV ANALYZER
# =====================================================
elif page == "📂 Bulk CSV Analyzer":
    st.markdown('<div class="section-title">📂 Bulk CSV / Real Dataset Analyzer</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3>📂 Upload Social Media Dataset</h3>
        <p>Upload tweets, YouTube comments, product reviews, customer feedback, or social media comments.</p>
        <p><b>Required:</b> Any one text column such as text, tweet, comment, comments, review, feedback, selected_text.</p>
        <p><b>Optional:</b> sentiment, platform, campaign, date columns.</p>
    </div>
    """, unsafe_allow_html=True)

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:
        raw_df = pd.read_csv(file)

        st.markdown("### 📌 Original Dataset Preview")
        st.dataframe(raw_df.head(10), use_container_width=True)

        normalized_df, error = normalize_real_dataset(raw_df)

        if error:
            st.error(error)
            st.write("Available columns:", list(raw_df.columns))
        else:
            analyzed = analyze_dataframe(normalized_df)

            if "Real_Sentiment" in normalized_df.columns:
                analyzed["Real_Sentiment"] = normalized_df["Real_Sentiment"].values

                matched = (analyzed["Sentiment"] == analyzed["Real_Sentiment"]).sum()
                total_labeled = analyzed["Real_Sentiment"].notna().sum()
                real_match_score = round((matched / total_labeled) * 100, 2) if total_labeled > 0 else 0

                st.markdown(f"""
                <div class="metric-card green">
                    <div class="metric-value">{real_match_score}%</div>
                    <div class="metric-label">Prediction Match With Real Labels</div>
                </div>
                """, unsafe_allow_html=True)

            st.session_state["data"] = analyzed

            st.success("✅ Real dataset sentiment analysis completed successfully!")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(f"""
                <div class="metric-card blue">
                    <div class="metric-value">{analyzed.shape[0]}</div>
                    <div class="metric-label">Total Rows</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card purple">
                    <div class="metric-value">{analyzed.shape[1]}</div>
                    <div class="metric-label">Total Columns</div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="metric-card yellow">
                    <div class="metric-value">CSV</div>
                    <div class="metric-label">Dataset Type</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### ✅ Final Analyzed Dataset")
            st.dataframe(analyzed.head(30), use_container_width=True)

            st.markdown("### 🥧 Uploaded Dataset Sentiment Distribution")
            fig_uploaded = px.pie(
                analyzed,
                names="Sentiment",
                hole=0.45,
                color="Sentiment",
                color_discrete_map=sentiment_colors
            )
            fig_uploaded.update_traces(textposition="inside", textinfo="percent+label")
            fig_uploaded.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_uploaded, use_container_width=True)

            st.markdown("### 🌈 Uploaded Dataset Category Distribution")
            category_df = analyzed["Category"].value_counts().reset_index()
            category_df.columns = ["Category", "Count"]

            fig_cat = px.bar(
                category_df,
                x="Category",
                y="Count",
                color="Category",
                text="Count",
                color_discrete_sequence=multi_colors
            )
            fig_cat.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cat, use_container_width=True)

            csv = analyzed.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Analyzed CSV",
                csv,
                "sentiment_analysis_results.csv",
                "text/csv"
            )


# =====================================================
# PREMIUM DASHBOARD
# =====================================================
elif page == "📈 Premium Dashboard":
    st.markdown('<div class="section-title">📈 Premium Sentiment Dashboard</div>', unsafe_allow_html=True)

    if "data" not in st.session_state:
        st.warning("Please load sample data or upload CSV first.")
    else:
        df = st.session_state["data"].copy()

        st.markdown("### 🎛️ Smart Filters")
        f1, f2, f3 = st.columns(3)

        with f1:
            selected_sentiment = st.multiselect(
                "Filter by Sentiment",
                options=df["Sentiment"].unique(),
                default=list(df["Sentiment"].unique())
            )

        with f2:
            selected_platform = st.multiselect(
                "Filter by Platform",
                options=df["platform"].unique(),
                default=list(df["platform"].unique())
            )

        with f3:
            selected_category = st.multiselect(
                "Filter by Category",
                options=df["Category"].unique(),
                default=list(df["Category"].unique())
            )

        df = df[
            (df["Sentiment"].isin(selected_sentiment)) &
            (df["platform"].isin(selected_platform)) &
            (df["Category"].isin(selected_category))
        ]

        total = len(df)
        positive = (df["Sentiment"] == "Positive").sum()
        negative = (df["Sentiment"] == "Negative").sum()
        neutral = (df["Sentiment"] == "Neutral").sum()
        health = brand_health(df)
        avg_conf = round(df["Confidence"].mean(), 2) if total > 0 else 0

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(f'<div class="metric-card blue"><div class="metric-value">{total}</div><div class="metric-label">Total Comments</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card green"><div class="metric-value">{positive}</div><div class="metric-label">Positive</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card red"><div class="metric-value">{negative}</div><div class="metric-label">Negative</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card purple"><div class="metric-value">{neutral}</div><div class="metric-label">Neutral</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="metric-card yellow"><div class="metric-value">{health}%</div><div class="metric-label">Brand Health</div></div>', unsafe_allow_html=True)

        st.markdown("### 🥧 Sentiment Distribution")
        fig1 = px.pie(
            df,
            names="Sentiment",
            hole=0.48,
            color="Sentiment",
            color_discrete_map=sentiment_colors
        )
        fig1.update_traces(textposition="inside", textinfo="percent+label")
        fig1.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("### 🌈 Colorful Category Distribution")
        category_count = df["Category"].value_counts().reset_index()
        category_count.columns = ["Category", "Count"]

        fig2 = px.bar(
            category_count,
            x="Category",
            y="Count",
            color="Category",
            color_discrete_sequence=multi_colors,
            text="Count"
        )
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 📊 Sentiment by Category")
        fig3 = px.histogram(
            df,
            x="Category",
            color="Sentiment",
            barmode="group",
            color_discrete_map=sentiment_colors
        )
        fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### 📈 Sentiment Trend Over Time")
        df["date"] = pd.to_datetime(df["date"])
        trend = df.groupby(["date", "Sentiment"]).size().reset_index(name="Count")

        fig4 = px.line(
            trend,
            x="date",
            y="Count",
            color="Sentiment",
            markers=True,
            color_discrete_map=sentiment_colors
        )
        fig4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("### 💡 Auto Business Insights")

        if health >= 40:
            st.markdown('<div class="insight">✅ Brand reputation is strong. Positive sentiment is clearly dominating.</div>', unsafe_allow_html=True)
        elif health >= 10:
            st.markdown('<div class="insight">⚠️ Brand sentiment is stable but should be monitored regularly.</div>', unsafe_allow_html=True)
        elif health >= 0:
            st.markdown('<div class="insight">⚠️ Brand sentiment is slightly positive, but negative feedback is meaningful.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight">🚨 Brand health is weak. Negative feedback needs urgent attention.</div>', unsafe_allow_html=True)

        if total > 0:
            top_category = df["Category"].value_counts().idxmax()
            st.markdown(f'<div class="insight">📍 Most discussed category: <b>{top_category}</b></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="insight">🧠 Average ML confidence score: <b>{avg_conf}%</b></div>', unsafe_allow_html=True)


# =====================================================
# KEYWORD INTELLIGENCE
# =====================================================
elif page == "🔍 Keyword Intelligence":
    st.markdown('<div class="section-title">🔍 Keyword Intelligence</div>', unsafe_allow_html=True)

    if "data" not in st.session_state:
        st.warning("Please load sample data or upload CSV first.")
    else:
        df = st.session_state["data"]

        keyword_df = top_keywords(df, 15)

        fig = px.bar(
            keyword_df,
            x="Keyword",
            y="Frequency",
            color="Keyword",
            color_discrete_sequence=multi_colors,
            text="Frequency",
            title="Most Repeated Customer Keywords"
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(keyword_df, use_container_width=True)


# =====================================================
# COMPLAINT MONITOR
# =====================================================
elif page == "🚨 Complaint Monitor":
    st.markdown('<div class="section-title">🚨 Complaint Monitor</div>', unsafe_allow_html=True)

    if "data" not in st.session_state:
        st.warning("Please load sample data or upload CSV first.")
    else:
        df = st.session_state["data"]

        complaints = df[df["Sentiment"] == "Negative"]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f'<div class="metric-card red"><div class="metric-value">{len(complaints)}</div><div class="metric-label">Total Complaints</div></div>', unsafe_allow_html=True)

        with c2:
            high_priority = (complaints["Priority"] == "High").sum()
            st.markdown(f'<div class="metric-card yellow"><div class="metric-value">{high_priority}</div><div class="metric-label">High Priority</div></div>', unsafe_allow_html=True)

        with c3:
            unique_categories = complaints["Category"].nunique()
            st.markdown(f'<div class="metric-card purple"><div class="metric-value">{unique_categories}</div><div class="metric-label">Issue Types</div></div>', unsafe_allow_html=True)

        if len(complaints) == 0:
            st.success("No negative complaints found.")
        else:
            st.markdown("### 📌 Complaint Table")
            st.dataframe(complaints, use_container_width=True)

            fig = px.histogram(
                complaints,
                x="Category",
                color="Priority",
                color_discrete_sequence=multi_colors,
                title="Complaint Categories by Priority"
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


# =====================================================
# CAMPAIGN ANALYTICS
# =====================================================
elif page == "📊 Campaign Analytics":
    st.markdown('<div class="section-title">📊 Campaign & Platform Analytics</div>', unsafe_allow_html=True)

    if "data" not in st.session_state:
        st.warning("Please load sample data or upload CSV first.")
    else:
        df = st.session_state["data"]

        st.markdown("### 📱 Platform-wise Sentiment")
        fig1 = px.histogram(
            df,
            x="platform",
            color="Sentiment",
            barmode="group",
            color_discrete_map=sentiment_colors
        )
        fig1.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("### 🎯 Campaign-wise Sentiment")
        fig2 = px.histogram(
            df,
            x="campaign",
            color="Sentiment",
            barmode="group",
            color_discrete_map=sentiment_colors
        )
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 🧩 Sentiment Share by Campaign")
        fig3 = px.sunburst(
            df,
            path=["campaign", "Sentiment"],
            color="Sentiment",
            color_discrete_map=sentiment_colors
        )
        fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)


# =====================================================
# ML MODEL INFO
# =====================================================
elif page == "🧠 ML Model Info":
    st.markdown('<div class="section-title">🧠 ML Model Comparison & Evaluation</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    best_acc = model_results_df.loc[model_results_df["Model"] == best_model_name, "Accuracy"].values[0]
    best_f1 = model_results_df.loc[model_results_df["Model"] == best_model_name, "F1 Score"].values[0]

    with c1:
        st.markdown(f"""
        <div class="metric-card green">
            <div class="metric-value">{best_model_name}</div>
            <div class="metric-label">Best Selected Model</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card blue">
            <div class="metric-value">{best_acc}%</div>
            <div class="metric-label">Best Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card purple">
            <div class="metric-value">{best_f1}%</div>
            <div class="metric-label">Best F1 Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Model Comparison Table")
    st.dataframe(model_results_df, use_container_width=True)

    st.markdown("### 🌈 Accuracy Comparison")
    fig_acc = px.bar(
        model_results_df,
        x="Model",
        y="Accuracy",
        color="Model",
        text="Accuracy",
        color_discrete_sequence=["#22C55E", "#3B82F6", "#EC4899"]
    )
    fig_acc.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("### 📈 F1 Score Comparison")
    fig_f1 = px.bar(
        model_results_df,
        x="Model",
        y="F1 Score",
        color="Model",
        text="F1 Score",
        color_discrete_sequence=["#F97316", "#A855F7", "#06B6D4"]
    )
    fig_f1.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_f1, use_container_width=True)

    st.markdown("### 🧮 Confusion Matrix")
    cm_df = pd.DataFrame(
        confusion_matrix_values,
        index=confusion_labels,
        columns=confusion_labels
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Turbo",
        labels=dict(x="Predicted Sentiment", y="Actual Sentiment", color="Count")
    )
    fig_cm.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("### 🧾 Model Pipeline Details")
    model_info = pd.DataFrame({
        "Stage": [
            "Raw Text Input",
            "Text Cleaning",
            "Feature Extraction",
            "Model 1",
            "Model 2",
            "Model 3",
            "Evaluation",
            "Final Prediction"
        ],
        "Details": [
            "Tweets, comments, reviews, or feedback text",
            "Lowercase, remove links, mentions, hashtags, symbols, and extra spaces",
            "TF-IDF Vectorizer with unigram and bigram features",
            "Logistic Regression",
            "Multinomial Naive Bayes",
            "Random Forest Classifier",
            "Accuracy, Weighted F1 Score, Confusion Matrix",
            "Positive, Negative, or Neutral"
        ]
    })

    st.dataframe(model_info, use_container_width=True)

# =====================================================
# DOWNLOAD REPORT
# =====================================================
elif page == "📤 Download Report":
    st.markdown('<div class="section-title">📤 Download Sentiment Report</div>', unsafe_allow_html=True)

    if "data" not in st.session_state:
        st.warning("Please load sample data or upload CSV first.")
    else:
        df = st.session_state["data"]

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Final Sentiment Intelligence Report",
            csv,
            "final_sentiment_intelligence_report.csv",
            "text/csv"
        )

        st.success("Your analyzed sentiment report is ready to download.")

st.markdown("""
<div class="footer">
Built for Data Science • NLP • Machine Learning • Business Analytics • Recruiter Portfolio 🚀
</div>
""", unsafe_allow_html=True)
