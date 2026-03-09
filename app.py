import json
import os
import re
import sys
import unittest
from collections import Counter
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None


DATASET_PATH = "/mnt/data/train_small_1000_en_LABELED_3THEMES_semantic_clean.csv"
DEFAULT_THEME_THRESHOLD = 0.32
HUMAN_REVIEW_THRESHOLD = 0.22
MAX_SELECTABLE_ROWS = 2000

THEME_LABELS = {
    "livraison": "Delivery",
    "sav": "Customer Support",
    "produit": "Product",
}

THEME_KEYWORDS = {
    "livraison": {
        "strong": [
            "delivery",
            "shipping",
            "shipment",
            "package",
            "parcel",
            "delivered",
            "arrived",
            "courier",
            "carrier",
            "tracking",
            "late delivery",
            "delayed delivery",
            "damaged package",
            "shipping delay",
        ],
        "medium": [
            "arrive",
            "arrival",
            "delayed",
            "delay",
            "late",
            "dispatch",
            "warehouse",
            "box",
            "packaging",
            "receive",
            "received",
        ],
    },
    "sav": {
        "strong": [
            "customer service",
            "customer support",
            "support team",
            "contacted support",
            "refund request",
            "return request",
            "no response",
            "never answered",
            "did not respond",
            "seller support",
        ],
        "medium": [
            "support",
            "refund",
            "return",
            "exchange",
            "response",
            "reply",
            "answered",
            "agent",
            "service",
            "warranty",
            "help desk",
            "customer care",
            "contact",
        ],
    },
    "produit": {
        "strong": [
            "product quality",
            "poor quality",
            "defective product",
            "broken product",
            "damaged item",
            "wrong size",
            "bad material",
            "cheap material",
        ],
        "medium": [
            "product",
            "quality",
            "material",
            "fabric",
            "size",
            "fit",
            "broken",
            "broke",
            "damaged",
            "defective",
            "scratchy",
            "thin",
            "cheap",
            "sturdy",
            "comfortable",
            "beautiful",
            "design",
            "color",
            "colour",
            "item",
        ],
    },
}

POSITIVE_TERMS = {
    "excellent",
    "great",
    "amazing",
    "perfect",
    "good",
    "love",
    "fast",
    "quick",
    "helpful",
    "resolved",
    "comfortable",
    "beautiful",
    "sturdy",
    "happy",
    "satisfied",
    "recommend",
    "smooth",
}

NEGATIVE_TERMS = {
    "bad",
    "poor",
    "terrible",
    "awful",
    "slow",
    "late",
    "delayed",
    "broken",
    "damaged",
    "defective",
    "cheap",
    "thin",
    "scratchy",
    "frustrating",
    "disappointed",
    "refund",
    "return",
    "problem",
    "issue",
    "never",
    "worse",
    "worst",
    "missing",
}

NEGATION_TERMS = {"not", "never", "no", "hardly", "barely", "n't"}
SENTIMENT_LABELS = {-1: "negative", 0: "neutral", 1: "positive"}


@dataclass
class ThemeResult:
    present: int
    sentiment: Optional[str]
    confidence: float
    evidence: List[str]


# =========================
# CORE ENGINE
# =========================
def normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text



def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", normalize_text(text))



def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text or ""))
    cleaned = [normalize_text(p) for p in parts if normalize_text(p)]
    return cleaned or [normalize_text(text)]



def term_present(sentence: str, term: str) -> bool:
    term = normalize_text(term)
    if " " in term:
        return term in sentence
    return re.search(rf"\b{re.escape(term)}\b", sentence) is not None



def collect_theme_evidence(text: str, theme: str) -> Tuple[float, List[str]]:
    sentences = split_sentences(text)
    strong_terms = THEME_KEYWORDS[theme]["strong"]
    medium_terms = THEME_KEYWORDS[theme]["medium"]

    score = 0.0
    evidence: List[str] = []

    for sentence in sentences:
        sentence_score = 0.0
        matched_terms: List[str] = []

        for term in strong_terms:
            if term_present(sentence, term):
                sentence_score += 0.28
                matched_terms.append(term)

        for term in medium_terms:
            if term_present(sentence, term):
                sentence_score += 0.12
                matched_terms.append(term)

        if sentence_score > 0:
            score += min(sentence_score, 0.55)
            evidence.extend(matched_terms[:3])

    score = min(score, 0.98)
    deduped_evidence = []
    for item in evidence:
        if item not in deduped_evidence:
            deduped_evidence.append(item)
    return score, deduped_evidence[:5]



def detect_themes(text: str, threshold: float) -> Dict[str, ThemeResult]:
    normalized = normalize_text(text)
    results: Dict[str, ThemeResult] = {}
    for theme in THEME_KEYWORDS:
        confidence, evidence = collect_theme_evidence(normalized, theme)
        present = int(confidence >= threshold)
        results[theme] = ThemeResult(
            present=present,
            sentiment=None,
            confidence=round(confidence, 2),
            evidence=evidence,
        )
    return results



def score_sentence_sentiment(sentence: str) -> Tuple[int, List[str], List[str]]:
    tokens = tokenize(sentence)
    if not tokens:
        return 0, [], []

    score = 0
    positives: List[str] = []
    negatives: List[str] = []

    for idx, token in enumerate(tokens):
        window = tokens[max(0, idx - 2): idx]
        negated = any(w in NEGATION_TERMS for w in window)

        if token in POSITIVE_TERMS:
            if negated:
                score -= 1
                negatives.append(f"not {token}")
            else:
                score += 1
                positives.append(token)
        elif token in NEGATIVE_TERMS:
            if negated:
                score += 1
                positives.append(f"not {token}")
            else:
                score -= 1
                negatives.append(token)

    return score, positives[:3], negatives[:3]



def score_sentiment(text: str) -> Tuple[str, float, List[str], List[str]]:
    sentences = split_sentences(text)
    total = 0
    pos_terms: List[str] = []
    neg_terms: List[str] = []

    for sentence in sentences:
        score, pos, neg = score_sentence_sentiment(sentence)
        total += score
        pos_terms.extend(pos)
        neg_terms.extend(neg)

    magnitude = abs(total)
    if total > 0:
        confidence = min(0.52 + 0.12 * magnitude, 0.95)
        return "positive", round(confidence, 2), pos_terms[:5], neg_terms[:5]
    if total < 0:
        confidence = min(0.52 + 0.12 * magnitude, 0.95)
        return "negative", round(confidence, 2), pos_terms[:5], neg_terms[:5]
    return "neutral", 0.5, pos_terms[:5], neg_terms[:5]



def extract_theme_context(text: str, theme: str) -> str:
    sentences = split_sentences(text)
    theme_terms = THEME_KEYWORDS[theme]["strong"] + THEME_KEYWORDS[theme]["medium"]
    matched = [s for s in sentences if any(term_present(s, term) for term in theme_terms)]
    if matched:
        return " ".join(matched[:2])
    return normalize_text(text)



def score_theme_sentiment(text: str, theme: str) -> Tuple[str, float, List[str], List[str]]:
    context = extract_theme_context(text, theme)
    return score_sentiment(context)



def human_review_needed(theme_results: Dict[str, ThemeResult], global_sentiment_confidence: float, original_text: str, threshold: float) -> bool:
    has_borderline_theme = any(HUMAN_REVIEW_THRESHOLD <= r.confidence < threshold for r in theme_results.values())
    short_text = len(tokenize(original_text)) < 4
    no_theme = not any(r.present == 1 for r in theme_results.values())
    uncertain_sentiment = global_sentiment_confidence < 0.56
    return has_borderline_theme or short_text or (no_theme and uncertain_sentiment)



def build_actionable_text(theme: str, sentiment: Optional[str]) -> str:
    mapping = {
        "livraison": {
            "negative": "Investigate shipping delay, carrier performance, and package condition.",
            "positive": "Highlight delivery reliability and speed in CX reporting.",
            "neutral": "Monitor delivery feedback for recurring logistics patterns.",
        },
        "sav": {
            "negative": "Escalate to support operations and review response/resolution time.",
            "positive": "Use this feedback as a proof point for customer support quality.",
            "neutral": "Monitor support interactions and identify weak spots in service flow.",
        },
        "produit": {
            "negative": "Review product quality, sizing, material, or catalog accuracy.",
            "positive": "Surface product strengths in merchandising and customer insights.",
            "neutral": "Track product mentions to refine product insight categories.",
        },
    }
    if sentiment not in {"negative", "positive", "neutral"}:
        sentiment = "neutral"
    return mapping.get(theme, {}).get(sentiment, "Monitor this topic.")



def analyze_review(text: str, review_id: str, threshold: float = DEFAULT_THEME_THRESHOLD) -> Dict:
    normalized = normalize_text(text)
    theme_results = detect_themes(normalized, threshold=threshold)
    global_sentiment, global_sentiment_confidence, pos_terms, neg_terms = score_sentiment(normalized)

    insights = []
    detected_themes: List[str] = []
    confidence_values: List[float] = []

    for theme, result in theme_results.items():
        if result.present == 1:
            detected_themes.append(theme)
            theme_sentiment, theme_sent_conf, _, _ = score_theme_sentiment(normalized, theme)
            combined_conf = round(min((result.confidence * 0.6) + (theme_sent_conf * 0.4), 0.98), 2)
            confidence_values.append(combined_conf)
            theme_results[theme] = ThemeResult(
                present=1,
                sentiment=theme_sentiment,
                confidence=combined_conf,
                evidence=result.evidence,
            )
            insights.append(
                {
                    "topic": theme,
                    "topic_label": THEME_LABELS[theme],
                    "sentiment": theme_sentiment,
                    "confidence": combined_conf,
                    "evidence": result.evidence,
                    "actionable_text": build_actionable_text(theme, theme_sentiment),
                }
            )

    if not insights:
        insights.append(
            {
                "topic": "other",
                "topic_label": "Other",
                "sentiment": global_sentiment,
                "confidence": global_sentiment_confidence,
                "evidence": pos_terms[:2] + neg_terms[:2],
                "actionable_text": "No strong theme was detected. Keep this review for taxonomy enrichment.",
            }
        )

    score_global = round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else global_sentiment_confidence
    needs_review = human_review_needed(theme_results, global_sentiment_confidence, normalized, threshold)

    payload = {
        "review_id": review_id,
        "review_text": text,
        "themes_detected": detected_themes,
        "insights": insights,
        "score_global": score_global,
        "global_sentiment": global_sentiment,
        "global_sentiment_confidence": global_sentiment_confidence,
        "positive_terms": pos_terms,
        "negative_terms": neg_terms,
        "needs_human_review": needs_review,
        "model_version": "english-poc-rules-v4",
    }

    for theme in THEME_KEYWORDS:
        result = theme_results[theme]
        payload[f"theme_{theme}"] = result.present
        payload[f"sent_{theme}"] = result.sentiment if result.present else None
        payload[f"conf_{theme}"] = result.confidence
        payload[f"evidence_{theme}"] = result.evidence

    return payload



def analyze_dataframe(df: pd.DataFrame, text_col: str, id_col: Optional[str], threshold: float) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' is missing from the dataset.")

    rows = []
    texts = df[text_col].fillna("").astype(str).tolist()
    ids = df[id_col].tolist() if id_col and id_col in df.columns else None

    for idx, text in enumerate(texts, start=1):
        review_id = str(ids[idx - 1]) if ids is not None else f"review_{idx}"
        rows.append(analyze_review(text, review_id, threshold=threshold))
    return pd.DataFrame(rows)



def flatten_export(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in results_df.to_dict(orient="records"):
        for insight in row.get("insights", []):
            rows.append(
                {
                    "review_id": row["review_id"],
                    "topic": insight["topic"],
                    "topic_label": insight["topic_label"],
                    "sentiment": insight["sentiment"],
                    "confidence": insight["confidence"],
                    "evidence": ", ".join(insight.get("evidence", [])),
                    "actionable_text": insight["actionable_text"],
                    "global_sentiment": row["global_sentiment"],
                    "needs_human_review": row["needs_human_review"],
                }
            )
    return pd.DataFrame(rows)



def safe_read_csv_filelike(file_obj) -> pd.DataFrame:
    try:
        return pd.read_csv(file_obj)
    except UnicodeDecodeError:
        file_obj.seek(0)
        return pd.read_csv(file_obj, encoding="latin-1")
    except Exception:
        file_obj.seek(0)
        return pd.read_csv(file_obj, sep=";")



def load_default_dataset() -> pd.DataFrame:
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame(
        {
            "review_id": ["R-001", "R-002", "R-003"],
            "review_body": [
                "The delivery was late and the package arrived damaged.",
                "Customer support never answered my refund request.",
                "The product looks great but the material feels cheap.",
            ],
            "sent_global": [-1, -1, -1],
            "theme_livraison": [1, 0, 0],
            "theme_sav": [0, 1, 0],
            "theme_produit": [0, 0, 1],
        }
    )



def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "review_body" not in prepared.columns:
        if "review_text" in prepared.columns:
            prepared["review_body"] = prepared["review_text"]
        elif "text" in prepared.columns:
            prepared["review_body"] = prepared["text"]
    if "review_title" not in prepared.columns:
        prepared["review_title"] = ""
    if "review_id" not in prepared.columns:
        prepared["review_id"] = [f"review_{i+1}" for i in range(len(prepared))]

    if "sent_global" in prepared.columns:
        prepared["sentiment_label"] = prepared["sent_global"].map(SENTIMENT_LABELS).fillna("neutral")
    else:
        prepared["sentiment_label"] = "unknown"

    for theme in THEME_KEYWORDS:
        col = f"theme_{theme}"
        if col not in prepared.columns:
            prepared[col] = 0

    return prepared



def find_text_column(df: pd.DataFrame) -> str:
    for col in ["review_body", "review_text", "text", "review_title"]:
        if col in df.columns:
            return col
    return df.columns[0]



def find_id_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["review_id", "id"]:
        if col in df.columns:
            return col
    return None



def filter_dataset(df: pd.DataFrame, query: str, sentiment_filter: str, theme_filter: str, text_col: str) -> pd.DataFrame:
    filtered = df.copy()

    if query.strip():
        q = query.strip().lower()
        searchable_cols = [c for c in ["review_id", "review_title", text_col] if c in filtered.columns]
        mask = pd.Series(False, index=filtered.index)
        for col in searchable_cols:
            mask = mask | filtered[col].fillna("").astype(str).str.lower().str.contains(q, regex=False)
        filtered = filtered[mask]

    if sentiment_filter != "All" and "sentiment_label" in filtered.columns:
        filtered = filtered[filtered["sentiment_label"] == sentiment_filter]

    theme_map = {
        "All": None,
        "Delivery": "theme_livraison",
        "Customer Support": "theme_sav",
        "Product": "theme_produit",
    }
    theme_col = theme_map.get(theme_filter)
    if theme_col and theme_col in filtered.columns:
        filtered = filtered[filtered[theme_col] == 1]

    return filtered.reset_index(drop=True)



def build_dashboard(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    sentiment_counts = df["sentiment_label"].value_counts().rename_axis("sentiment").to_frame("count")
    theme_counts = pd.DataFrame(
        {
            "theme": ["Delivery", "Customer Support", "Product"],
            "count": [
                int(df["theme_livraison"].sum()),
                int(df["theme_sav"].sum()),
                int(df["theme_produit"].sum()),
            ],
        }
    )
    return {"sentiment_counts": sentiment_counts, "theme_counts": theme_counts}


# =========================
# UI
# =========================
def configure_page() -> None:
    if st is None:
        return
    st.set_page_config(page_title="Review Intelligence POC", page_icon="💬", layout="wide", initial_sidebar_state="expanded")



def inject_styles() -> None:
    if st is None:
        return
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f5f7fb 0%, #edf2f8 100%);
            color: #162033;
        }
        .block-container {
            max-width: 1420px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid rgba(0,0,0,0.06);
        }
        .hero {
            background: linear-gradient(135deg, #162033 0%, #24344d 70%, #35527a 100%);
            color: white;
            border-radius: 24px;
            padding: 1.5rem 1.6rem;
            box-shadow: 0 18px 50px rgba(18, 32, 51, 0.18);
            margin-bottom: 1rem;
        }
        .hero h1 { margin: 0; font-size: 2.2rem; }
        .hero p { margin-top: 0.6rem; color: rgba(255,255,255,0.88); }
        .card {
            background: white;
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 0 10px 25px rgba(18, 32, 51, 0.06);
            margin-bottom: 1rem;
        }
        .theme-card {
            background: white;
            border-radius: 18px;
            padding: 1rem;
            border: 1px solid rgba(0,0,0,0.06);
            min-height: 210px;
            box-shadow: 0 8px 20px rgba(18, 32, 51, 0.05);
        }
        .theme-positive { border-left: 6px solid #16a34a; }
        .theme-negative { border-left: 6px solid #dc2626; }
        .theme-neutral { border-left: 6px solid #64748b; }
        .theme-off { border-left: 6px solid #e2e8f0; opacity: 0.8; }
        .badge {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-right: 0.4rem;
            margin-bottom: 0.35rem;
        }
        .badge-positive { background: #dcfce7; color: #166534; }
        .badge-negative { background: #fee2e2; color: #991b1b; }
        .badge-neutral { background: #e2e8f0; color: #334155; }
        .badge-theme { background: #dbeafe; color: #1d4ed8; }
        .section-title { font-size: 1.05rem; font-weight: 800; margin-bottom: 0.6rem; }
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 18px;
            padding: 0.7rem 0.8rem;
            box-shadow: 0 8px 18px rgba(18, 32, 51, 0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



def sentiment_badge(sentiment: str) -> str:
    mapping = {
        "positive": '<span class="badge badge-positive">Positive</span>',
        "negative": '<span class="badge badge-negative">Negative</span>',
        "neutral": '<span class="badge badge-neutral">Neutral</span>',
        "unknown": '<span class="badge badge-neutral">Unknown</span>',
    }
    return mapping.get(sentiment, mapping["neutral"])



def theme_badge(label: str) -> str:
    return f'<span class="badge badge-theme">{label}</span>'



def render_header(df: pd.DataFrame) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div style="font-size:0.82rem; letter-spacing:0.15em; text-transform:uppercase; opacity:0.78; font-weight:700;">English Review Intelligence POC</div>
            <h1>Theme and sentiment detection for customer reviews</h1>
            <p>Upload a CSV, browse the full dataset, select any review, and get a clear theme and sentiment readout instantly. The current version is optimized for English reviews only.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows in dataset", len(df))
    with c2:
        st.metric("Delivery labels", int(df["theme_livraison"].sum()))
    with c3:
        st.metric("Support labels", int(df["theme_sav"].sum()))
    with c4:
        st.metric("Product labels", int(df["theme_produit"].sum()))



def render_analysis_summary(result: Dict) -> None:
    theme_html = "".join(theme_badge(THEME_LABELS[t]) for t in result["themes_detected"]) if result["themes_detected"] else theme_badge("Other")
    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">Instant result</div>
            <div style="margin-bottom:0.45rem;">{sentiment_badge(result['global_sentiment'])}{theme_html}</div>
            <div>Confidence score: <b>{result['score_global']}</b></div>
            <div>Human review needed: <b>{'Yes' if result['needs_human_review'] else 'No'}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_theme_cards(result: Dict) -> None:
    cols = st.columns(3)
    for idx, theme in enumerate(["livraison", "sav", "produit"]):
        present = result.get(f"theme_{theme}", 0) == 1
        sentiment = result.get(f"sent_{theme}") or "not detected"
        confidence = result.get(f"conf_{theme}", 0)
        evidence = result.get(f"evidence_{theme}", [])

        css = "theme-off"
        if present and sentiment == "positive":
            css = "theme-positive"
        elif present and sentiment == "negative":
            css = "theme-negative"
        elif present:
            css = "theme-neutral"

        evidence_text = ", ".join(evidence) if evidence else "No direct evidence found"
        action_text = build_actionable_text(theme, result.get(f"sent_{theme}")) if present else "No strong signal for this theme in the review."

        cols[idx].markdown(
            f"""
            <div class="theme-card {css}">
                <div style="font-size:1.08rem; font-weight:800;">{THEME_LABELS[theme]}</div>
                <div style="margin-top:0.45rem;"><b>Detected:</b> {'Yes' if present else 'No'}</div>
                <div><b>Sentiment:</b> {sentiment}</div>
                <div><b>Confidence:</b> {confidence}</div>
                <div style="margin-top:0.5rem;"><b>Evidence:</b> {evidence_text}</div>
                <div style="margin-top:0.65rem; color:#475569;">{action_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )



def render_dataset_preview(df: pd.DataFrame) -> None:
    cols = [c for c in ["review_id", "review_title", "review_body", "sentiment_label", "theme_livraison", "theme_sav", "theme_produit"] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, height=720)



def run_streamlit_app() -> None:
    if st is None:
        raise RuntimeError("Streamlit is not available in this environment.")

    configure_page()
    inject_styles()

    with st.sidebar:
        st.markdown("## Dataset")
        uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
        threshold = st.slider("Theme detection threshold", 0.15, 0.85, DEFAULT_THEME_THRESHOLD, 0.05)

    raw_df = safe_read_csv_filelike(uploaded_file) if uploaded_file is not None else load_default_dataset()
    df = prepare_dataset(raw_df)
    text_col_default = find_text_column(df)
    id_col_default = find_id_column(df)

    with st.sidebar:
        st.markdown("## Column mapping")
        text_col = st.selectbox("Review text column", options=list(df.columns), index=list(df.columns).index(text_col_default))
        id_choices = [""] + list(df.columns)
        id_index = id_choices.index(id_col_default) if id_col_default in id_choices else 0
        id_col = st.selectbox("Review ID column", options=id_choices, index=id_index)
        id_col = id_col or None
        st.markdown("## Current source")
        st.write(f"Rows: {len(df)}")
        st.write(f"Source: {'Uploaded CSV' if uploaded_file is not None else 'Default dataset'}")

    render_header(df)
    dashboard = build_dashboard(df)

    tab1, tab2, tab3 = st.tabs(["Instant analysis", "Full dataset", "POC dashboard"])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Analyze a review</div>', unsafe_allow_html=True)
        st.write("You can type your own review or select any row from the uploaded dataset.")

        f1, f2, f3 = st.columns([1.1, 0.8, 0.8])
        with f1:
            query = st.text_input("Search in the dataset", value="")
        with f2:
            sentiment_filter = st.selectbox("Ground-truth sentiment filter", ["All", "positive", "negative", "neutral", "unknown"], index=0)
        with f3:
            theme_filter = st.selectbox("Ground-truth theme filter", ["All", "Delivery", "Customer Support", "Product"], index=0)

        filtered_df = filter_dataset(df, query, sentiment_filter, theme_filter, text_col)
        st.caption(f"{len(filtered_df)} matching row(s) available for analysis.")

        preview_cols = [c for c in [id_col, "review_title", text_col, "sentiment_label"] if c and c in filtered_df.columns]
        if preview_cols:
            st.dataframe(filtered_df[preview_cols], use_container_width=True, height=360)

        selection_options = ["Manual review input"]
        lookup: Dict[str, Dict] = {}
        for idx, row in filtered_df.head(MAX_SELECTABLE_ROWS).iterrows():
            row_id = str(row[id_col]) if id_col and id_col in filtered_df.columns else f"row_{idx+1}"
            preview = str(row[text_col])[:140].replace("\n", " ")
            label = f"{row_id} — {preview}"
            selection_options.append(label)
            lookup[label] = row.to_dict()

        selected = st.selectbox("Select a dataset row to analyze", selection_options, index=0)
        selected_text = ""
        selected_id = "demo_review_001"
        ground_truth = {}
        if selected != "Manual review input":
            selected_row = lookup[selected]
            selected_text = str(selected_row[text_col])
            selected_id = str(selected_row[id_col]) if id_col and id_col in selected_row else "dataset_row"
            ground_truth = {
                "sentiment": selected_row.get("sentiment_label", "unknown"),
                "delivery": selected_row.get("theme_livraison", 0),
                "support": selected_row.get("theme_sav", 0),
                "product": selected_row.get("theme_produit", 0),
            }

        left, right = st.columns([1.0, 1.0])
        with left:
            review_text = st.text_area(
                "Review text",
                value=selected_text,
                height=240,
                placeholder="Type an English review here or select one from the dataset above.",
            )
            review_id = st.text_input("Review ID", value=selected_id)
        with right:
            effective_text = review_text if str(review_text).strip() else selected_text
            result = analyze_review(effective_text, review_id, threshold=threshold)
            render_analysis_summary(result)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Operational readout</div>', unsafe_allow_html=True)
            if result["needs_human_review"]:
                st.warning("This review is borderline or ambiguous and should be checked by a human.")
            else:
                st.success("This review is clear enough for quick operational reading.")
            st.write(f"Positive clues: {', '.join(result['positive_terms']) if result['positive_terms'] else 'None'}")
            st.write(f"Negative clues: {', '.join(result['negative_terms']) if result['negative_terms'] else 'None'}")
            st.markdown('</div>', unsafe_allow_html=True)

        render_theme_cards(result)

        if ground_truth:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Ground truth from dataset</div>', unsafe_allow_html=True)
            gt_cols = st.columns(4)
            with gt_cols[0]:
                st.metric("Sentiment label", ground_truth["sentiment"])
            with gt_cols[1]:
                st.metric("Delivery label", int(ground_truth["delivery"]))
            with gt_cols[2]:
                st.metric("Support label", int(ground_truth["support"]))
            with gt_cols[3]:
                st.metric("Product label", int(ground_truth["product"]))
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Show analysis JSON"):
            st.json(result)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Full dataset</div>', unsafe_allow_html=True)
        st.caption(f"Complete dataset currently loaded in the app: {len(df)} row(s).")
        render_dataset_preview(df)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Batch re-analysis using the POC logic</div>', unsafe_allow_html=True)
        results_df = analyze_dataframe(df, text_col=text_col, id_col=id_col, threshold=threshold)
        export_df = flatten_export(results_df)
        st.dataframe(export_df, use_container_width=True, height=720)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download enriched CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="review_intelligence_poc.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "Download analysis JSON",
                data=results_df.to_json(orient="records", force_ascii=False, indent=2),
                file_name="review_intelligence_poc.json",
                mime="application/json",
                use_container_width=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Dataset rows", len(df))
        with k2:
            st.metric("Negative labels", int((df["sentiment_label"] == "negative").sum()))
        with k3:
            st.metric("Positive labels", int((df["sentiment_label"] == "positive").sum()))
        with k4:
            st.metric("Neutral labels", int((df["sentiment_label"] == "neutral").sum()))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Theme distribution</div>', unsafe_allow_html=True)
            st.bar_chart(dashboard["theme_counts"].set_index("theme"))
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Sentiment distribution</div>', unsafe_allow_html=True)
            st.bar_chart(dashboard["sentiment_counts"])
            st.markdown('</div>', unsafe_allow_html=True)


# =========================
# CLI FALLBACK
# =========================
def run_cli_demo() -> None:
    df = prepare_dataset(load_default_dataset())
    text_col = find_text_column(df)
    example = str(df[text_col].iloc[0]) if len(df) else "The delivery was late and the package arrived damaged."
    print(json.dumps(analyze_review(example, "cli_demo_001"), ensure_ascii=False, indent=2))


# =========================
# TESTS
# =========================
class ReviewIntelligenceTests(unittest.TestCase):
    def test_delivery_theme_detection_in_english(self):
        result = detect_themes("the delivery was late and the package arrived damaged", 0.32)
        self.assertEqual(result["livraison"].present, 1)

    def test_support_theme_detection_in_english(self):
        result = detect_themes("customer support never answered my refund request", 0.32)
        self.assertEqual(result["sav"].present, 1)

    def test_product_theme_detection_in_english(self):
        result = detect_themes("the product looks nice but the material is cheap and thin", 0.32)
        self.assertEqual(result["produit"].present, 1)

    def test_positive_sentiment(self):
        label, confidence, _, _ = score_sentiment("great quality and very comfortable")
        self.assertEqual(label, "positive")
        self.assertGreater(confidence, 0.5)

    def test_negative_sentiment(self):
        label, confidence, _, _ = score_sentiment("terrible support and bad refund process")
        self.assertEqual(label, "negative")
        self.assertGreater(confidence, 0.5)

    def test_prepare_dataset_generates_ids(self):
        df = pd.DataFrame({"review_body": ["hello"]})
        prepared = prepare_dataset(df)
        self.assertIn("review_id", prepared.columns)
        self.assertEqual(prepared.iloc[0]["review_id"], "review_1")

    def test_find_text_column_prefers_review_body(self):
        df = pd.DataFrame({"review_body": ["a"], "text": ["b"]})
        self.assertEqual(find_text_column(df), "review_body")

    def test_filter_dataset_by_query(self):
        df = prepare_dataset(pd.DataFrame({
            "review_id": ["1", "2"],
            "review_body": ["late delivery", "great product"],
            "sent_global": [-1, 1],
            "theme_livraison": [1, 0],
            "theme_sav": [0, 0],
            "theme_produit": [0, 1],
        }))
        filtered = filter_dataset(df, "delivery", "All", "All", "review_body")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["review_id"], "1")

    def test_analyze_review_returns_theme_and_sentiment(self):
        result = analyze_review("customer support was slow and never answered", "r1")
        self.assertIn("sav", result["themes_detected"])
        self.assertEqual(result["global_sentiment"], "negative")

    def test_safe_read_csv_semicolon(self):
        csv_content = StringIO("review_id;review_body\n1;late delivery")
        df = safe_read_csv_filelike(csv_content)
        self.assertIn("review_body", df.columns)


if __name__ == "__main__":
    if "--test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    elif st is None:
        run_cli_demo()
    else:
        run_streamlit_app()
