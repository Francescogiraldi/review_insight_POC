import json
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
DEFAULT_CONFIDENCE_THRESHOLD = 0.60
HUMAN_REVIEW_THRESHOLD = 0.45


@dataclass
class ThemeResult:
    present: int
    sentiment: Optional[str]
    confidence: float


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
        results[theme] = ThemeResult(
            present=present,
            sentiment=None,
            confidence=round(confidence, 2),
        )
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
            "livraison": "Prioriser une alerte logistique et vérifier les délais / dommages colis.",
            "sav": "Escalader au support client et suivre le temps de résolution.",
            "produit": "Analyser le défaut produit et ouvrir une boucle qualité / catalogue.",
        }
        return mapping.get(theme, "Déclencher une revue opérationnelle.")
    if sentiment == "positif":
        return f"Capitaliser sur les retours positifs liés à {theme} dans les insights CX."
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
                "actionable_text": "Classer en backlog pour enrichir la taxonomie métier.",
            }
        )

    score_global = round(sum(sentiment_scores) / len(sentiment_scores), 2) if sentiment_scores else 0.50

    payload = {
        "review_id": review_id,
        "review_text": text,
        "themes_detected": detected_themes,
        "insights": insights,
        "score_global": score_global,
        "needs_human_review": needs_review,
        "model_version": "poc-rules-v2-premium-ui",
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



def build_sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["R-001", "R-002", "R-003", "R-004", "R-005", "R-006"],
            "review_text": [
                "La livraison est arrivée en retard et le carton était abîmé.",
                "Le produit est excellent, très bonne qualité.",
                "Le service client n'a jamais répondu à ma demande de remboursement.",
                "Produit correct mais support lent et retour compliqué.",
                "Livraison rapide et produit conforme, super expérience.",
                "Le colis est arrivé, mais le produit semble défectueux.",
            ],
        }
    )



def safe_read_csv_filelike(file_obj):
    try:
        return pd.read_csv(file_obj)
    except UnicodeDecodeError:
        file_obj.seek(0)
        return pd.read_csv(file_obj, encoding="latin-1")
    except Exception:
        file_obj.seek(0)
        return pd.read_csv(file_obj, sep=";")



def configure_page() -> None:
    if st is None:
        return
    st.set_page_config(
        page_title="Review Insights+ | MVP Studio",
        page_icon="✨",
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
                    radial-gradient(circle at top left, rgba(110, 87, 224, 0.22), transparent 28%),
                    radial-gradient(circle at top right, rgba(17, 153, 142, 0.16), transparent 24%),
                    linear-gradient(180deg, #0b1020 0%, #11182d 52%, #0d1325 100%);
                color: #f4f7fb;
            }
            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2rem;
                max-width: 1440px;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(10,16,32,0.96), rgba(16,24,45,0.98));
                border-right: 1px solid rgba(255,255,255,0.08);
            }
            [data-testid="stMetric"] {
                background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.06));
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 18px;
                padding: 14px 16px;
                backdrop-filter: blur(10px);
            }
            div[data-testid="stMetricValue"] { color: #ffffff; }
            div[data-testid="stMetricLabel"] { color: #aab6d3; }
            .hero-card {
                padding: 1.45rem 1.5rem;
                border-radius: 24px;
                border: 1px solid rgba(255,255,255,0.08);
                background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.05));
                box-shadow: 0 24px 80px rgba(0,0,0,0.28);
                backdrop-filter: blur(18px);
                margin-bottom: 1rem;
            }
            .glass-card {
                padding: 1rem 1.1rem;
                border-radius: 20px;
                border: 1px solid rgba(255,255,255,0.08);
                background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
                box-shadow: 0 18px 42px rgba(0,0,0,0.18);
                margin-bottom: 1rem;
            }
            .section-title {
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                color: #8ea0c9;
                margin-bottom: 0.35rem;
                font-weight: 700;
            }
            .headline {
                font-size: 2.2rem;
                font-weight: 800;
                line-height: 1.1;
                color: #ffffff;
                margin-bottom: 0.4rem;
            }
            .subtle { color: #b7c2da; font-size: 1rem; }
            .pill {
                display: inline-block;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                margin-right: 0.45rem;
                margin-bottom: 0.45rem;
                font-size: 0.84rem;
                font-weight: 600;
                border: 1px solid rgba(255,255,255,0.08);
                background: rgba(255,255,255,0.06);
                color: #e8ecf8;
            }
            .insight-positive,
            .insight-negative,
            .insight-neutral {
                border-radius: 18px;
                padding: 1rem 1rem;
                margin-bottom: 0.75rem;
                border: 1px solid rgba(255,255,255,0.10);
                box-shadow: 0 14px 34px rgba(0,0,0,0.16);
            }
            .insight-positive { background: linear-gradient(135deg, rgba(25,135,84,0.24), rgba(255,255,255,0.04)); }
            .insight-negative { background: linear-gradient(135deg, rgba(220,53,69,0.26), rgba(255,255,255,0.04)); }
            .insight-neutral { background: linear-gradient(135deg, rgba(13,110,253,0.24), rgba(255,255,255,0.04)); }
            .mini-kpi {
                border-radius: 16px;
                padding: 0.9rem 1rem;
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.07);
                margin-bottom: 0.8rem;
            }
            .mini-kpi-label {
                color: #93a5cc;
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
            }
            .mini-kpi-value {
                color: #ffffff;
                font-size: 1.5rem;
                font-weight: 800;
                margin-top: 0.15rem;
            }
            .streamlit-expanderHeader { background: rgba(255,255,255,0.04); border-radius: 14px; }
            .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
            .stTabs [data-baseweb="tab"] {
                background: rgba(255,255,255,0.05);
                border-radius: 12px;
                padding: 0.55rem 1rem;
                color: #dfe7fb;
            }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, rgba(110,87,224,0.40), rgba(45,128,255,0.32));
            }
            .footer-note { text-align: center; color: #91a1c7; padding: 0.4rem 0 0.8rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )



def metric_card(label: str, value, help_text: Optional[str] = None) -> None:
    st.metric(label=label, value=value, help=help_text)



def mini_kpi(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="mini-kpi">
            <div class="mini-kpi-label">{label}</div>
            <div class="mini-kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="section-title">Customer Intelligence Platform</div>
            <div class="headline">Review Insights+ — MVP Studio</div>
            <div class="subtle">Un prototype premium pour transformer des avis clients en signaux actionnables, priorités opérationnelles et exports prêts à brancher dans une stack IA de production.</div>
            <div style="margin-top: 1rem;">
                <span class="pill">Multi-thèmes</span>
                <span class="pill">Sentiment par insight</span>
                <span class="pill">Human review queue</span>
                <span class="pill">Dashboard exécutif</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_single_result(result: Dict) -> None:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"### Analyse premium — {result['review_id']}")
    st.write(result["review_text"])

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Score global", result["score_global"])
    with col2:
        metric_card("Thèmes détectés", len(result["themes_detected"]))
    with col3:
        metric_card("Revue humaine", "Oui" if result["needs_human_review"] else "Non")

    st.markdown("#### Insights actionnables")
    for insight in result["insights"]:
        sentiment = insight["sentiment"] or "neutre"
        css_class = "insight-neutral"
        if sentiment == "négatif":
            css_class = "insight-negative"
        elif sentiment == "positif":
            css_class = "insight-positive"
        icon = "🧭"
        if sentiment == "négatif":
            icon = "🚨"
        elif sentiment == "positif":
            icon = "✨"

        st.markdown(
            f"""
            <div class="{css_class}">
                <div style="font-weight: 800; color: #ffffff; margin-bottom: 0.25rem;">{icon} {insight['topic'].title()}</div>
                <div style="color: #d9e2f7; margin-bottom: 0.35rem;">Sentiment: <b>{sentiment}</b> · Confiance: <b>{insight['confidence']}</b></div>
                <div style="color: #f2f5fc;">{insight['actionable_text']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Voir le payload JSON"):
        st.json(result)
    st.markdown("</div>", unsafe_allow_html=True)



def render_architecture_block() -> None:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Architecture cible</div>
            <div style="font-size: 1.08rem; color: #f4f7fb; line-height: 1.7;">
                1. Entrée texte ou batch CSV.<br>
                2. Détection multi-label des thèmes métier.<br>
                3. Sentiment conditionné par thème détecté.<br>
                4. Guardrails de confiance et file de revue humaine.<br>
                5. Sorties dashboard, alerting, export JSON/CSV et API-ready.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def run_streamlit_app() -> None:
    if st is None:
        raise RuntimeError("Streamlit n'est pas disponible dans cet environnement.")

    configure_page()
    inject_styles()
    render_header()

    with st.sidebar:
        st.markdown("## Command Center")
        threshold = st.slider("Seuil de détection thème", 0.30, 0.90, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)
        st.markdown("### Modules activés")
        st.write("- Détection thèmes")
        st.write("- Sentiment")
        st.write("- Alertes métier")
        st.write("- Export JSON / CSV")
        st.markdown("### Taxonomie")
        st.write("- Livraison")
        st.write("- SAV")
        st.write("- Produit")
        st.markdown("### Positionnement")
        st.caption(
            "Cette V2 privilégie une expérience direction produit / innovation, avec une UI premium et une structure plus prête pour une future industrialisation."
        )

    summary_results = analyze_dataframe(build_sample_data(), text_col="review_text", id_col="review_id", threshold=threshold)
    summary_export = flatten_export(summary_results)

    hero_c1, hero_c2, hero_c3, hero_c4 = st.columns(4)
    with hero_c1:
        mini_kpi("Avis monitorés", str(len(summary_results)))
    with hero_c2:
        mini_kpi("Alertes critiques", str(int((summary_export["sentiment"] == "négatif").sum())))
    with hero_c3:
        mini_kpi("Human review", str(int(summary_results["needs_human_review"].sum())))
    with hero_c4:
        mini_kpi("Score moyen", f"{summary_results['score_global'].mean():.2f}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Analyse premium", "Batch Studio", "Executive Dashboard", "Architecture & API"]
    )

    with tab1:
        left, right = st.columns([1.25, 0.75])
        with left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Analyse unitaire")
            sample_text = st.text_area(
                "Collez un avis client",
                value="J'ai un problème avec la livraison, le carton était abîmé. Le service client n'a pas pu m'aider rapidement.",
                height=170,
            )
            review_id = st.text_input("ID avis", value="demo_001")
            run_single = st.button("Analyser l'avis", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Signal design</div>
                    <div style="font-size: 1.05rem; color: #ffffff; font-weight: 700; margin-bottom: 0.45rem;">Ce que montre cette vue</div>
                    <div class="subtle">Une lecture rapide d'un avis avec niveau de confiance, thèmes détectés et recommandation métier directement exploitable en POC client ou comité produit.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            positive_count = int((summary_export["sentiment"] == "positif").sum())
            negative_count = int((summary_export["sentiment"] == "négatif").sum())
            neutral_count = int(summary_export["sentiment"].fillna("neutre").eq("neutre").sum())
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Snapshot")
            metric_card("Positifs", positive_count)
            metric_card("Négatifs", negative_count)
            metric_card("Neutres", neutral_count)
            st.markdown("</div>", unsafe_allow_html=True)

        if run_single:
            result = analyze_review(sample_text, review_id, threshold=threshold)
            render_single_result(result)

    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Batch Studio")
        uploaded_file = st.file_uploader("Uploader un CSV", type=["csv"])
        use_sample = st.checkbox("Utiliser un jeu d'exemple", value=True)
        st.markdown("</div>", unsafe_allow_html=True)

        source_df = None
        if uploaded_file is not None:
            try:
                source_df = safe_read_csv_filelike(uploaded_file)
            except Exception as exc:
                st.error(f"Impossible de lire le CSV : {exc}")
        elif use_sample:
            source_df = build_sample_data()

        if source_df is not None and not source_df.empty:
            preview_col, control_col = st.columns([1.3, 0.7])
            with preview_col:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### Aperçu des données")
                st.dataframe(source_df, use_container_width=True, height=320)
                st.markdown("</div>", unsafe_allow_html=True)
            with control_col:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### Mapping des colonnes")
                default_text_index = 1 if len(source_df.columns) > 1 else 0
                text_col = st.selectbox("Colonne contenant les avis", options=list(source_df.columns), index=default_text_index)
                id_index = list(source_df.columns).index("review_id") + 1 if "review_id" in source_df.columns else 0
                id_col = st.selectbox("Colonne ID (optionnel)", options=[""] + list(source_df.columns), index=id_index)
                run_batch = st.button("Lancer l'analyse batch", use_container_width=True, type="primary")
                st.markdown("</div>", unsafe_allow_html=True)

            if run_batch:
                results_df = analyze_dataframe(source_df, text_col=text_col, id_col=id_col or None, threshold=threshold)
                export_df = flatten_export(results_df)
                st.session_state["results_df"] = results_df
                st.session_state["export_df"] = export_df
                st.success("Analyse batch terminée.")

        if "results_df" in st.session_state:
            results_df = st.session_state["results_df"]
            export_df = st.session_state["export_df"]
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Résultats enrichis")
            st.dataframe(export_df, use_container_width=True, height=320)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label="Télécharger CSV enrichi",
                    data=export_df.to_csv(index=False).encode("utf-8"),
                    file_name="review_insights_export_v2.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                st.download_button(
                    label="Télécharger JSON",
                    data=results_df.to_json(orient="records", force_ascii=False, indent=2),
                    file_name="review_insights_payload_v2.json",
                    mime="application/json",
                    use_container_width=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        if "results_df" not in st.session_state:
            st.session_state["results_df"] = summary_results
            st.session_state["export_df"] = summary_export

        results_df = st.session_state["results_df"]
        export_df = st.session_state["export_df"]

        top_left, top_right = st.columns([1.05, 0.95])
        with top_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Executive KPIs")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                metric_card("Avis analysés", len(results_df))
            with k2:
                metric_card("Alertes négatives", int((export_df["sentiment"] == "négatif").sum()))
            with k3:
                metric_card("Revue humaine", int(results_df["needs_human_review"].sum()))
            with k4:
                metric_card("Score moyen", round(results_df["score_global"].mean(), 2))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Répartition par thème")
            theme_counts = pd.DataFrame(
                {
                    "theme": list(THEMES.keys()),
                    "count": [int(results_df[f"theme_{theme}"].sum()) for theme in THEMES],
                }
            )
            st.bar_chart(theme_counts.set_index("theme"))
            st.markdown("</div>", unsafe_allow_html=True)

        with top_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Répartition des sentiments")
            sentiment_counts = (
                export_df["sentiment"].fillna("non classé").value_counts().rename_axis("sentiment").to_frame("count")
            )
            st.bar_chart(sentiment_counts)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Alert queue")
            alerts_df = export_df[
                (export_df["sentiment"] == "négatif") | (export_df["needs_human_review"] == True)
            ].copy()
            alerts_df = alerts_df.sort_values(by=["needs_human_review", "confidence"], ascending=[False, True])
            st.dataframe(alerts_df, use_container_width=True, height=300)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        left, right = st.columns([1.05, 0.95])
        with left:
            render_architecture_block()
        with right:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">API payload preview</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.code(
                json.dumps(
                    {
                        "review_id": "12345",
                        "themes_detected": ["livraison", "sav"],
                        "insights": [
                            {
                                "topic": "livraison",
                                "sentiment": "négatif",
                                "actionable_text": "Prioriser une alerte logistique.",
                            },
                            {
                                "topic": "sav",
                                "sentiment": "négatif",
                                "actionable_text": "Escalader au support client.",
                            },
                        ],
                        "score_global": 0.74,
                        "needs_human_review": True,
                        "model_version": "poc-rules-v2-premium-ui",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                language="json",
            )
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Roadmap</div>
                    <div class="subtle">Étape suivante : brancher un backend FastAPI, substituer le moteur de règles par un classifieur multi-label Hugging Face et exposer des endpoints batch/inference temps réel.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="footer-note">
            V2 premium conçue pour une démo plus crédible en rendez-vous client, comité innovation ou validation de MVP data/IA.
        </div>
        """,
        unsafe_allow_html=True,
    )



def run_cli_demo() -> None:
    print("Streamlit n'est pas installé. Exécution du mode CLI de secours.\n")
    demo_result = analyze_review(
        "La livraison est arrivée en retard et le carton était abîmé.",
        "demo_cli_001",
        threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    print(json.dumps(demo_result, ensure_ascii=False, indent=2))
    print("\nPour lancer l'interface web, installez Streamlit puis exécutez :")
    print("streamlit run review_insights_poc_streamlit_app.py")


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

    def test_cli_demo_returns_none(self):
        self.assertIsNone(run_cli_demo())

    def test_detect_themes_respects_high_threshold(self):
        result = detect_themes("support", 0.95)
        self.assertEqual(result["sav"].present, 0)


if __name__ == "__main__":
    if "--test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    elif st is None:
        run_cli_demo()
    else:
        run_streamlit_app()
