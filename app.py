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
DEFAULT_CONFIDENCE_THRESHOLD = 0.60
HUMAN_REVIEW_THRESHOLD = 0.45

THEMES = {
    "livraison": [
        "livraison",
        "delivery",
        "retard",
        "late",
        "colis",
        "package",
        "carton",
        "expédition",
        "shipping",
        "arrivé",
        "arrive",
        "transport",
        "reçu",
    ],
    "sav": [
        "sav",
        "service client",
        "support",
        "customer service",
        "help",
        "assistance",
        "remboursement",
        "refund",
        "retour",
        "return",
        "réponse",
        "response",
    ],
    "produit": [
        "produit",
        "product",
        "qualité",
        "quality",
        "cassé",
        "broken",
        "défectueux",
        "defective",
        "taille",
        "size",
        "matière",
        "material",
        "couleur",
        "color",
    ],
}

POSITIVE_WORDS = {
    "excellent",
    "parfait",
    "super",
    "rapide",
    "satisfait",
    "top",
    "bonne",
    "bon",
    "great",
    "amazing",
    "good",
    "love",
    "resolved",
    "helpful",
}

NEGATIVE_WORDS = {
    "lent",
    "slow",
    "retard",
    "mauvais",
    "nul",
    "problème",
    "problem",
    "cassé",
    "broken",
    "déçu",
    "décevant",
    "frustrant",
    "frustrating",
    "jamais",
    "poor",
    "bad",
    "late",
    "damaged",
}

RARE_THEMES = {"livraison", "sav"}
THEME_LABELS = {
    "livraison": "Livraison",
    "sav": "SAV",
    "produit": "Produit",
}
SENTIMENT_LABELS = {
    -1: "négatif",
    0: "neutre",
    1: "positif",
}


@dataclass
class ThemeResult:
    present: int
    sentiment: Optional[str]
    confidence: float


# =========================
# MOTEUR D'ANALYSE POC
# =========================
def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text



def keyword_hits(text: str, keywords: List[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)



def detect_themes(text: str, threshold: float) -> Dict[str, ThemeResult]:
    results: Dict[str, ThemeResult] = {}
    for theme, keywords in THEMES.items():
        hits = keyword_hits(text, keywords)
        confidence = min(0.2 + hits * 0.22, 0.98) if hits > 0 else 0.08
        present = int(confidence >= threshold)
        results[theme] = ThemeResult(present=present, sentiment=None, confidence=round(confidence, 2))
    return results



def score_sentiment(text: str) -> Tuple[str, float]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts = Counter(tokens)
    pos = sum(counts[w] for w in POSITIVE_WORDS if w in counts)
    neg = sum(counts[w] for w in NEGATIVE_WORDS if w in counts)

    raw = pos - neg
    total = pos + neg

    if total == 0:
        return "neutre", 0.50

    confidence = min(0.55 + (abs(raw) / max(total, 1)) * 0.35, 0.95)
    if raw > 0:
        return "positif", round(confidence, 2)
    if raw < 0:
        return "négatif", round(confidence, 2)
    return "neutre", 0.55



def enrich_results(text: str, theme_results: Dict[str, ThemeResult]) -> Dict[str, ThemeResult]:
    sentiment, sent_conf = score_sentiment(text)
    enriched: Dict[str, ThemeResult] = {}
    for theme, result in theme_results.items():
        if result.present == 1:
            enriched[theme] = ThemeResult(
                present=1,
                sentiment=sentiment,
                confidence=round((result.confidence + sent_conf) / 2, 2),
            )
        else:
            enriched[theme] = result
    return enriched



def human_review_needed(theme_results: Dict[str, ThemeResult], original_text: str, threshold: float) -> bool:
    low_conf_theme = any(HUMAN_REVIEW_THRESHOLD <= r.confidence < threshold for r in theme_results.values())
    empty_short_text = len(original_text.split()) < 4
    rare_theme_low_margin = any(
        theme in RARE_THEMES and 0.45 <= result.confidence <= 0.65
        for theme, result in theme_results.items()
    )
    return low_conf_theme or empty_short_text or rare_theme_low_margin



def build_actionable_text(theme: str, sentiment: Optional[str]) -> str:
    if sentiment == "négatif":
        mapping = {
            "livraison": "Alerter la logistique et vérifier délai, transporteur et état du colis.",
            "sav": "Escalader au support client et suivre le temps de résolution.",
            "produit": "Ouvrir une boucle qualité pour défaut produit, fiche article ou packaging.",
        }
        return mapping.get(theme, "Déclencher une revue opérationnelle.")
    if sentiment == "positif":
        return f"Valoriser les retours positifs liés au thème {theme} dans les insights CX."
    return f"Surveiller le thème {theme} sans action urgente."



def analyze_review(text: str, review_id: str, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> Dict:
    normalized = normalize_text(text)
    themes = detect_themes(normalized, threshold=threshold)
    enriched = enrich_results(normalized, themes)
    needs_review = human_review_needed(themes, normalized, threshold=threshold)

    insights = []
    sentiment_scores = []
    detected_themes = []

    for theme, result in enriched.items():
        if result.present == 1:
            detected_themes.append(theme)
            sentiment_scores.append(result.confidence)
            insights.append(
                {
                    "topic": theme,
                    "sentiment": result.sentiment,
                    "confidence": result.confidence,
                    "actionable_text": build_actionable_text(theme, result.sentiment),
                }
            )

    if not detected_themes:
        global_sentiment, sent_conf = score_sentiment(normalized)
        insights.append(
            {
                "topic": "autre",
                "sentiment": global_sentiment,
                "confidence": sent_conf,
                "actionable_text": "Classer cet avis dans le backlog pour enrichir la taxonomie métier.",
            }
        )

    score_global = round(sum(sentiment_scores) / len(sentiment_scores), 2) if sentiment_scores else 0.50
    global_sentiment = insights[0]["sentiment"] if insights else "neutre"

    payload = {
        "review_id": review_id,
        "review_text": text,
        "themes_detected": detected_themes,
        "insights": insights,
        "score_global": score_global,
        "global_sentiment": global_sentiment,
        "needs_human_review": needs_review,
        "model_version": "poc-fr-v3-ux-ui",
    }

    for theme in THEMES:
        result = enriched[theme]
        payload[f"theme_{theme}"] = result.present
        payload[f"sent_{theme}"] = result.sentiment if result.present else None
        payload[f"conf_{theme}"] = result.confidence

    return payload



def analyze_dataframe(df: pd.DataFrame, text_col: str, id_col: Optional[str], threshold: float) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"La colonne texte '{text_col}' est absente du DataFrame.")

    rows = []
    text_values = df[text_col].fillna("").astype(str).tolist()
    id_values = df[id_col].tolist() if id_col and id_col in df.columns else None

    for idx, text in enumerate(text_values, start=1):
        review_id = str(id_values[idx - 1]) if id_values is not None else f"review_{idx}"
        result = analyze_review(text, review_id, threshold=threshold)
        rows.append(result)

    return pd.DataFrame(rows)



def flatten_export(results_df: pd.DataFrame) -> pd.DataFrame:
    export_rows = []
    for row in results_df.to_dict(orient="records"):
        insights = row.get("insights", [])
        if insights:
            for insight in insights:
                export_rows.append(
                    {
                        "review_id": row["review_id"],
                        "topic": insight["topic"],
                        "sentiment": insight["sentiment"],
                        "confidence": insight["confidence"],
                        "actionable_text": insight["actionable_text"],
                        "needs_human_review": row["needs_human_review"],
                        "score_global": row["score_global"],
                    }
                )
        else:
            export_rows.append(
                {
                    "review_id": row["review_id"],
                    "topic": None,
                    "sentiment": None,
                    "confidence": None,
                    "actionable_text": None,
                    "needs_human_review": row["needs_human_review"],
                    "score_global": row["score_global"],
                }
            )
    return pd.DataFrame(export_rows)



def safe_read_csv_filelike(file_obj):
    try:
        return pd.read_csv(file_obj)
    except UnicodeDecodeError:
        file_obj.seek(0)
        return pd.read_csv(file_obj, encoding="latin-1")
    except Exception:
        file_obj.seek(0)
        return pd.read_csv(file_obj, sep=";")



def find_text_column(df: pd.DataFrame) -> str:
    priorities = ["review_body", "review_text", "text_norm", "review_title", "text"]
    for col in priorities:
        if col in df.columns:
            return col
    return df.columns[0]



def find_id_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["review_id", "id"]:
        if col in df.columns:
            return col
    return None



def load_poc_dataset() -> pd.DataFrame:
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame(
        {
            "review_id": ["R-001", "R-002", "R-003"],
            "review_body": [
                "The delivery was late and the package was damaged.",
                "The product is excellent and the quality is great.",
                "Customer service never answered my refund request.",
            ],
            "sent_global": [-1, 1, -1],
            "theme_livraison": [1, 0, 0],
            "theme_sav": [0, 0, 1],
            "theme_produit": [0, 1, 0],
        }
    )



def prepare_demo_dataset(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "review_body" not in prepared.columns and "review_text" in prepared.columns:
        prepared["review_body"] = prepared["review_text"]
    if "review_body" not in prepared.columns and "text" in prepared.columns:
        prepared["review_body"] = prepared["text"]
    if "review_title" not in prepared.columns:
        prepared["review_title"] = ""
    if "review_id" not in prepared.columns:
        prepared["review_id"] = [f"review_{i+1}" for i in range(len(prepared))]
    if "sent_global" in prepared.columns:
        prepared["sentiment_fr"] = prepared["sent_global"].map(SENTIMENT_LABELS).fillna("neutre")
    else:
        prepared["sentiment_fr"] = "neutre"
    for theme in THEMES:
        theme_col = f"theme_{theme}"
        if theme_col not in prepared.columns:
            prepared[theme_col] = 0
    return prepared


# =========================
# UI STREAMLIT
# =========================
def configure_page() -> None:
    if st is None:
        return
    st.set_page_config(
        page_title="POC Analyse des Avis Clients",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )



def inject_styles() -> None:
    if st is None:
        return
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(88, 97, 246, 0.16), transparent 26%),
                    radial-gradient(circle at top right, rgba(16, 185, 129, 0.14), transparent 24%),
                    linear-gradient(180deg, #f7f9fc 0%, #edf2f8 100%);
                color: #142033;
            }
            .block-container {
                max-width: 1380px;
                padding-top: 1rem;
                padding-bottom: 2rem;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #ffffff 0%, #f5f8fc 100%);
                border-right: 1px solid rgba(20, 32, 51, 0.08);
            }
            .hero {
                padding: 1.6rem 1.6rem;
                border-radius: 26px;
                background: linear-gradient(135deg, #162033 0%, #24344d 60%, #324b6b 100%);
                color: white;
                box-shadow: 0 20px 60px rgba(19, 31, 49, 0.22);
                margin-bottom: 1rem;
            }
            .hero-kicker {
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.16em;
                opacity: 0.8;
                font-weight: 700;
            }
            .hero-title {
                font-size: 2.3rem;
                font-weight: 800;
                margin-top: 0.4rem;
                line-height: 1.08;
            }
            .hero-sub {
                margin-top: 0.65rem;
                color: rgba(255,255,255,0.86);
                font-size: 1rem;
                max-width: 900px;
            }
            .surface-card {
                background: rgba(255,255,255,0.9);
                border: 1px solid rgba(17,24,39,0.08);
                box-shadow: 0 10px 30px rgba(18, 33, 57, 0.08);
                border-radius: 22px;
                padding: 1rem 1rem;
                margin-bottom: 1rem;
            }
            .theme-card {
                border-radius: 20px;
                padding: 1rem 1rem;
                border: 1px solid rgba(17,24,39,0.08);
                background: white;
                box-shadow: 0 8px 24px rgba(18, 33, 57, 0.06);
                min-height: 170px;
            }
            .theme-ok {
                border-left: 6px solid #10b981;
            }
            .theme-bad {
                border-left: 6px solid #ef4444;
            }
            .theme-neutral {
                border-left: 6px solid #94a3b8;
            }
            .theme-off {
                border-left: 6px solid #e2e8f0;
                opacity: 0.76;
            }
            .badge {
                display: inline-block;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                font-weight: 700;
                font-size: 0.84rem;
                margin-right: 0.45rem;
                margin-bottom: 0.45rem;
            }
            .badge-pos { background: #dcfce7; color: #166534; }
            .badge-neg { background: #fee2e2; color: #991b1b; }
            .badge-neu { background: #e2e8f0; color: #334155; }
            .badge-theme { background: #dbeafe; color: #1d4ed8; }
            .section-title {
                font-size: 1.05rem;
                font-weight: 800;
                color: #162033;
                margin-bottom: 0.7rem;
            }
            .helper {
                color: #5b6b81;
                font-size: 0.96rem;
            }
            div[data-testid="stMetric"] {
                background: white;
                border: 1px solid rgba(17,24,39,0.08);
                border-radius: 18px;
                padding: 0.75rem 0.85rem;
                box-shadow: 0 8px 24px rgba(18, 33, 57, 0.05);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )



def sentiment_badge(sentiment: str) -> str:
    if sentiment == "positif":
        return '<span class="badge badge-pos">Positif</span>'
    if sentiment == "négatif":
        return '<span class="badge badge-neg">Négatif</span>'
    return '<span class="badge badge-neu">Neutre</span>'



def theme_badge(label: str) -> str:
    return f'<span class="badge badge-theme">{label}</span>'



def render_header(df_demo: pd.DataFrame) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">POC / MVP en français</div>
            <div class="hero-title">Analyse des commentaires clients par thème et sentiment</div>
            <div class="hero-sub">Écrivez un commentaire et obtenez immédiatement une lecture claire : thème détecté, sentiment, niveau de confiance et action recommandée. Le POC s'appuie aussi sur le fichier fourni pour la démonstration et le dashboard.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    a, b, c, d = st.columns(4)
    with a:
        st.metric("Commentaires du fichier", len(df_demo))
    with b:
        st.metric("Commentaires livraison", int(df_demo["theme_livraison"].sum()))
    with c:
        st.metric("Commentaires SAV", int(df_demo["theme_sav"].sum()))
    with d:
        st.metric("Commentaires produit", int(df_demo["theme_produit"].sum()))



def render_analysis_summary(result: Dict) -> None:
    themes = result["themes_detected"] or ["autre"]
    badges = "".join(theme_badge(THEME_LABELS.get(t, t.title())) for t in themes)
    st.markdown(
        f"""
        <div class="surface-card">
            <div class="section-title">Résultat instantané</div>
            <div style="margin-bottom:0.5rem;">{sentiment_badge(result['global_sentiment'])}{badges}</div>
            <div class="helper">Score global : <b>{result['score_global']}</b> · Revue humaine : <b>{'Oui' if result['needs_human_review'] else 'Non'}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_theme_cards(result: Dict) -> None:
    cols = st.columns(3)
    for idx, theme in enumerate(["livraison", "sav", "produit"]):
        theme_label = THEME_LABELS[theme]
        is_present = result[f"theme_{theme}"] == 1
        sentiment = result.get(f"sent_{theme}") or "non détecté"
        confidence = result.get(f"conf_{theme}", 0)
        action = build_actionable_text(theme, result.get(f"sent_{theme}")) if is_present else "Aucun signal fort détecté sur ce thème."

        css_class = "theme-off"
        if is_present and sentiment == "positif":
            css_class = "theme-ok"
        elif is_present and sentiment == "négatif":
            css_class = "theme-bad"
        elif is_present:
            css_class = "theme-neutral"

        cols[idx].markdown(
            f"""
            <div class="theme-card {css_class}">
                <div style="font-size:1.08rem;font-weight:800;color:#142033;">{theme_label}</div>
                <div style="margin-top:0.45rem;color:#334155;"><b>Détection :</b> {'Oui' if is_present else 'Non'}</div>
                <div style="margin-top:0.25rem;color:#334155;"><b>Sentiment :</b> {sentiment}</div>
                <div style="margin-top:0.25rem;color:#334155;"><b>Confiance :</b> {confidence}</div>
                <div style="margin-top:0.7rem;color:#475569;line-height:1.5;">{action}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )



def render_dataset_preview(df_demo: pd.DataFrame) -> None:
    preview_cols = [c for c in ["review_id", "review_title", "review_body", "sentiment_fr", "theme_livraison", "theme_sav", "theme_produit"] if c in df_demo.columns]
    st.dataframe(df_demo[preview_cols], use_container_width=True, height=700)



def build_dashboard_from_uploaded(df_demo: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    sentiment_counts = df_demo["sentiment_fr"].value_counts().rename_axis("sentiment").to_frame("count")
    theme_counts = pd.DataFrame(
        {
            "thème": ["Livraison", "SAV", "Produit"],
            "volume": [
                int(df_demo["theme_livraison"].sum()),
                int(df_demo["theme_sav"].sum()),
                int(df_demo["theme_produit"].sum()),
            ],
        }
    )
    return {
        "sentiment_counts": sentiment_counts,
        "theme_counts": theme_counts,
    }


def filter_dataset(df_demo: pd.DataFrame, query: str, sentiment_filter: str, theme_filter: str, text_col: str) -> pd.DataFrame:
    filtered = df_demo.copy()

    if query.strip():
        q = query.strip().lower()
        searchable_cols = [c for c in ["review_id", "review_title", text_col] if c in filtered.columns]
        mask = pd.Series(False, index=filtered.index)
        for col in searchable_cols:
            mask = mask | filtered[col].fillna("").astype(str).str.lower().str.contains(q, regex=False)
        filtered = filtered[mask]

    if sentiment_filter != "Tous" and "sentiment_fr" in filtered.columns:
        filtered = filtered[filtered["sentiment_fr"] == sentiment_filter]

    theme_map = {
        "Tous": None,
        "Livraison": "theme_livraison",
        "SAV": "theme_sav",
        "Produit": "theme_produit",
    }
    theme_col = theme_map.get(theme_filter)
    if theme_col and theme_col in filtered.columns:
        filtered = filtered[filtered[theme_col] == 1]

    return filtered.reset_index(drop=True)


def load_uploaded_or_default_dataset(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return safe_read_csv_filelike(uploaded_file)
    return load_poc_dataset()



def run_streamlit_app() -> None:
    if st is None:
        raise RuntimeError("Streamlit n'est pas disponible dans cet environnement.")

    configure_page()
    inject_styles()

    with st.sidebar:
        st.markdown("## Paramètres du POC")
        uploaded_file = st.file_uploader("Importer un dataset CSV", type=["csv"])
        threshold = st.slider("Seuil de détection des thèmes", 0.30, 0.90, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)

        df_raw = load_uploaded_or_default_dataset(uploaded_file)
        df_demo = prepare_demo_dataset(df_raw)
        available_columns = list(df_demo.columns)
        detected_text_col = find_text_column(df_demo)
        detected_id_col = find_id_column(df_demo)

        text_col = st.selectbox(
            "Colonne du commentaire à analyser",
            options=available_columns,
            index=available_columns.index(detected_text_col) if detected_text_col in available_columns else 0,
        )
        id_options = [""] + available_columns
        id_default = id_options.index(detected_id_col) if detected_id_col in id_options else 0
        id_col = st.selectbox("Colonne ID", options=id_options, index=id_default)
        id_col = id_col or None

        dashboard = build_dashboard_from_uploaded(df_demo)

        st.markdown("### Dataset utilisé")
        st.caption("Vous pouvez importer votre propre CSV. Le fichier importé alimente l'analyse, le tableau et le dashboard.")
        st.write(f"- Source : {'CSV importé' if uploaded_file is not None else 'Dataset par défaut'}")
        st.write(f"- Lignes : {len(df_demo)}")
        st.write(f"- Colonne texte : {text_col}")
        st.write(f"- Colonne ID : {id_col or 'générée'}")
        st.markdown("### Logique métier")
        st.write("- Détection des thèmes")
        st.write("- Sentiment associé")
        st.write("- File de revue humaine")
        st.write("- Export opérationnel")

    render_header(df_demo)

    tab1, tab2, tab3 = st.tabs(["Analyse instantanée", "Commentaires du fichier", "Dashboard POC"])

    with tab1:
        st.markdown('<div class="surface-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Analyse instantanée</div>', unsafe_allow_html=True)
        st.markdown('<div class="helper">Écrivez un commentaire libre ou sélectionnez une ligne du dataset importé pour l’analyser immédiatement.</div>', unsafe_allow_html=True)

        filter_col1, filter_col2, filter_col3 = st.columns([1.2, 0.8, 0.8])
        with filter_col1:
            search_query = st.text_input("Rechercher dans le dataset", value="")
        with filter_col2:
            sentiment_filter = st.selectbox("Filtrer par sentiment", options=["Tous", "positif", "négatif", "neutre"], index=0)
        with filter_col3:
            theme_filter = st.selectbox("Filtrer par thème", options=["Tous", "Livraison", "SAV", "Produit"], index=0)

        filtered_df = filter_dataset(df_demo, search_query, sentiment_filter, theme_filter, text_col)
        preview_cols = [c for c in [id_col, "review_title", text_col, "sentiment_fr"] if c and c in filtered_df.columns]
        if preview_cols:
            st.dataframe(filtered_df[preview_cols], use_container_width=True, height=380)

        selection_options = ["Commentaire libre"]
        selected_lookup = {}
        for idx, row in filtered_df.iterrows():
            rid = str(row[id_col]) if id_col and id_col in filtered_df.columns else f"row_{idx + 1}"
            preview = str(row[text_col])[:100].replace("\n", " ")
            label = f"{rid} — {preview}"
            selection_options.append(label)
            selected_lookup[label] = row

        selected_source = st.selectbox("Choisir un commentaire du dataset", options=selection_options, index=0)
        default_text = ""
        default_review_id = "demo_live_001"
        if selected_source != "Commentaire libre":
            selected_row = selected_lookup[selected_source]
            default_text = str(selected_row[text_col])
            default_review_id = str(selected_row[id_col]) if id_col and id_col in filtered_df.columns else "demo_dataset_001"

        left, right = st.columns([1.05, 0.95])
        with left:
            live_text = st.text_area(
                "Commentaire client",
                value=default_text,
                height=220,
                placeholder="Écrivez ici un commentaire ou utilisez une ligne du dataset importé.",
            )
            review_id = st.text_input("ID du commentaire", value=default_review_id)

        with right:
            effective_text = live_text if str(live_text).strip() else default_text
            result = analyze_review(effective_text, review_id, threshold=threshold)
            render_analysis_summary(result)
            st.markdown('<div class="surface-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Lecture rapide</div>', unsafe_allow_html=True)
            st.markdown(
                "<div class='helper'>Le commentaire sélectionné ou saisi est analysé immédiatement selon la logique du POC : thème, sentiment, confiance et action recommandée.</div>",
                unsafe_allow_html=True,
            )
            if result["needs_human_review"]:
                st.warning("Ce commentaire mérite une revue humaine pour sécuriser la décision métier.")
            else:
                st.success("Le signal est suffisamment clair pour une lecture métier rapide.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        render_theme_cards(result)

        with st.expander("Voir le JSON du POC"):
            st.json(result)

    with tab2:
        st.markdown('<div class="surface-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset complet utilisé dans le POC</div>', unsafe_allow_html=True)
        st.caption(f"Nombre total de lignes du fichier : {len(df_demo)}")
        st.markdown('<div class="helper">Le CSV importé est visible ici en entier et sert aussi de base pour l’analyse instantanée.</div>', unsafe_allow_html=True)
        render_dataset_preview(df_demo)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="surface-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Ré-analyser tout le dataset avec la logique POC</div>', unsafe_allow_html=True)
        results_df = analyze_dataframe(df_demo, text_col=text_col, id_col=id_col, threshold=threshold)
        export_df = flatten_export(results_df)
        st.dataframe(export_df, use_container_width=True, height=700)
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                label="Télécharger le CSV enrichi",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="poc_analyse_commentaires.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_b:
            st.download_button(
                label="Télécharger le JSON du POC",
                data=results_df.to_json(orient="records", force_ascii=False, indent=2),
                file_name="poc_analyse_commentaires.json",
                mime="application/json",
                use_container_width=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="surface-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Répartition par thème dans le fichier</div>', unsafe_allow_html=True)
            st.bar_chart(dashboard["theme_counts"].set_index("thème"))
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="surface-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Répartition des sentiments du fichier</div>', unsafe_allow_html=True)
            st.bar_chart(dashboard["sentiment_counts"])
            st.markdown('</div>', unsafe_allow_html=True)

        if "review_body" in df_demo.columns:
            alert_mask = (
                (df_demo.get("theme_livraison", 0) == 1)
                | (df_demo.get("theme_sav", 0) == 1)
                | (df_demo.get("theme_produit", 0) == 1)
            )
            alert_cols = [c for c in ["review_id", "review_body", "sentiment_fr", "theme_livraison", "theme_sav", "theme_produit"] if c in df_demo.columns]
            st.markdown('<div class="surface-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Commentaires à fort intérêt métier</div>', unsafe_allow_html=True)
            st.dataframe(df_demo.loc[alert_mask, alert_cols].head(30), use_container_width=True, height=360)
            st.markdown('</div>', unsafe_allow_html=True)


# =========================
# MODE CLI DE SECOURS
# =========================
def run_cli_demo() -> None:
    df_raw = load_poc_dataset()
    df_demo = prepare_demo_dataset(df_raw)
    text_col = find_text_column(df_demo)
    review_text = str(df_demo[text_col].iloc[0]) if len(df_demo) else "The delivery was late and damaged."
    print("Streamlit n'est pas installé. Exécution du mode CLI du POC.\n")
    print(json.dumps(analyze_review(review_text, "demo_cli_001"), ensure_ascii=False, indent=2))


# =========================
# TESTS
# =========================
class ReviewInsightsTests(unittest.TestCase):
    def test_normalize_text_collapses_spaces(self):
        self.assertEqual(normalize_text("  Bonjour   le   monde  "), "bonjour le monde")

    def test_detect_themes_finds_livraison(self):
        result = detect_themes("livraison en retard avec colis abîmé", 0.60)
        self.assertEqual(result["livraison"].present, 1)

    def test_score_sentiment_negative(self):
        sentiment, confidence = score_sentiment("retard cassé mauvais")
        self.assertEqual(sentiment, "négatif")
        self.assertGreaterEqual(confidence, 0.55)

    def test_analyze_review_returns_autre_when_no_theme(self):
        result = analyze_review("Très bien", "r1")
        self.assertEqual(result["insights"][0]["topic"], "autre")

    def test_analyze_review_exposes_global_sentiment(self):
        result = analyze_review("excellent produit", "r2")
        self.assertIn("global_sentiment", result)

    def test_analyze_dataframe_generates_default_ids(self):
        df = pd.DataFrame({"text": ["excellent produit", "service client lent"]})
        results = analyze_dataframe(df, text_col="text", id_col=None, threshold=0.60)
        self.assertEqual(results["review_id"].tolist(), ["review_1", "review_2"])

    def test_analyze_dataframe_raises_on_missing_text_col(self):
        df = pd.DataFrame({"body": ["test"]})
        with self.assertRaises(ValueError):
            analyze_dataframe(df, text_col="text", id_col=None, threshold=0.60)

    def test_flatten_export_produces_rows(self):
        df = pd.DataFrame([analyze_review("produit cassé", "r2")])
        flat = flatten_export(df)
        self.assertGreaterEqual(len(flat), 1)
        self.assertIn("topic", flat.columns)

    def test_safe_read_csv_filelike_supports_semicolon_separator(self):
        csv_content = StringIO("review_id;review_text\nR-1;Livraison rapide")
        df = safe_read_csv_filelike(csv_content)
        self.assertEqual(df.columns.tolist(), ["review_id", "review_text"])
        self.assertEqual(df.iloc[0]["review_id"], "R-1")

    def test_find_text_column_prefers_review_body(self):
        df = pd.DataFrame({"review_body": ["a"], "text": ["b"]})
        self.assertEqual(find_text_column(df), "review_body")

    def test_prepare_demo_dataset_adds_missing_theme_columns(self):
        df = pd.DataFrame({"review_text": ["abc"]})
        prepared = prepare_demo_dataset(df)
        self.assertIn("theme_livraison", prepared.columns)
        self.assertIn("theme_sav", prepared.columns)
        self.assertIn("theme_produit", prepared.columns)

    def test_load_poc_dataset_returns_dataframe(self):
        df = load_poc_dataset()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_prepare_demo_dataset_generates_review_id(self):
        df = pd.DataFrame({"review_body": ["abc"]})
        prepared = prepare_demo_dataset(df)
        self.assertIn("review_id", prepared.columns)
        self.assertEqual(prepared.iloc[0]["review_id"], "review_1")

    def test_filter_dataset_filters_text(self):
        df = pd.DataFrame(
            {
                "review_id": ["1", "2"],
                "review_body": ["late delivery", "great product"],
                "sentiment_fr": ["négatif", "positif"],
                "theme_livraison": [1, 0],
                "theme_sav": [0, 0],
                "theme_produit": [0, 1],
            }
        )
        filtered = filter_dataset(df, "delivery", "Tous", "Tous", "review_body")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["review_id"], "1")


if __name__ == "__main__":
    if "--test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    elif st is None:
        run_cli_demo()
    else:
        run_streamlit_app()
