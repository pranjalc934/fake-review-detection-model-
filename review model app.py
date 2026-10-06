"""
Review Checker - an accessible Streamlit front end for the fake-review detector
(CountVectorizer + Multinomial Naive Bayes) from the Colab notebook.

SETUP (everything is in this one file)
    1. pip install streamlit pandas numpy scikit-learn matplotlib
    2. Put your dataset (archive.zip, or a .csv with `text_` and `label` columns)
       in the same folder as this file, or upload it inside the app.
    3. streamlit run review_checker_app.py

Labels: in the common fake-reviews dataset, CG = computer-generated (fake),
OR = original (genuine).
"""

import html
import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

APP_DIR = Path(__file__).parent

# Theme defaults (replaces a .streamlit/config.toml file). Best effort: if this
# Streamlit version refuses, the custom CSS below still styles the page.
for _k, _v in {
    "theme.base": "light",
    "theme.primaryColor": "#0E6A78",
    "theme.backgroundColor": "#F2F7F8",
    "theme.secondaryBackgroundColor": "#E3EFF2",
    "theme.textColor": "#10263B",
}.items():
    try:
        st._config.set_option(_k, _v)
    except Exception:  # noqa: BLE001
        pass

st.set_page_config(
    page_title="Review Checker",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Colour palettes (all text/background pairs meet WCAG AA, 4.5:1 or better)
# ----------------------------------------------------------------------------
PALETTES = {
    "Standard": dict(
        bg="#F2F7F8", surface="#FFFFFF", side="#E3EFF2", ink="#10263B", muted="#44566A",
        primary="#0E6A78", on_primary="#FFFFFF", soft="#DCEEF1", border="#8CA7B5",
        good="#0F6B43", good_bg="#E3F4EB", bad="#A8261C", bad_bg="#FCE8E6",
        warn="#7A4E00", warn_bg="#FFF3D6", focus="#5B3FD0",
    ),
    "High contrast": dict(
        bg="#000000", surface="#000000", side="#000000", ink="#FFFFFF", muted="#FFFFFF",
        primary="#FFD60A", on_primary="#000000", soft="#1A1A1A", border="#FFFFFF",
        good="#7CFFB2", good_bg="#000000", bad="#FF9E96", bad_bg="#000000",
        warn="#FFD60A", warn_bg="#000000", focus="#00E5FF",
    ),
}

FONTS = {
    "Atkinson Hyperlegible (clear letters, great for low vision)": (
        "'Atkinson Hyperlegible', 'Segoe UI', Arial, sans-serif",
        "family=Atkinson+Hyperlegible:wght@400;700",
    ),
    "Lexend (wider spacing, easier for dyslexia)": (
        "'Lexend', 'Segoe UI', Arial, sans-serif",
        "family=Lexend:wght@400;600",
    ),
    "System default": ("system-ui, 'Segoe UI', Arial, sans-serif", None),
}

TEXT_SIZES = {"Normal": 112.5, "Large": 137.5, "Extra large": 175.0}  # % of 16px

EXAMPLES = {
    "Enthusiastic": "This product is amazing and I absolutely love it!",
    "Negative": "Worst product ever, completely useless and not worth the money.",
    "Detailed": "Excellent quality, arrived on time and works perfectly.",
}

SURE_ENOUGH = 0.65  # below this confidence we say "Not sure"


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
def build_css(p, size_pct, font_family, font_import, extra_spacing):
    spacing = (
        "line-height:1.9 !important; letter-spacing:0.04em; word-spacing:0.14em;"
        if extra_spacing
        else "line-height:1.65;"
    )
    imp = f"@import url('https://fonts.googleapis.com/css2?{font_import}&display=swap');" if font_import else ""
    css = """
    %IMPORT%
    html { font-size: %SIZE%%; }
    .stApp, .stApp p, .stApp li, .stApp label, .stApp textarea, .stApp input,
    .stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp [data-baseweb="tab"], .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricLabel"], .stApp table, .stApp th, .stApp td {
        font-family: %FONT% !important; %SPACING%
    }
    .stApp { background: %bg%; color: %ink%; }
    [data-testid="stHeader"] { background: %bg%; }
    [data-testid="stSidebar"] { background: %side%; border-right: 2px solid %border%; }
    .stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp [data-testid="stMarkdownContainer"] { color: %ink%; }
    .stApp .stCaption, .stApp [data-testid="stCaptionContainer"] { color: %muted%; }
    h1 { font-weight: 700 !important; letter-spacing: -0.01em; }
    .block-container { max-width: 880px; padding-top: 2.2rem; }

    /* Inputs */
    .stApp textarea, .stApp input[type="text"], .stApp [data-baseweb="select"] > div {
        background: %surface% !important; color: %ink% !important;
        border: 2px solid %border% !important; border-radius: 10px !important;
    }
    .stApp textarea { min-height: 9rem; }

    /* Buttons: big targets, clear labels */
    .stApp .stButton > button { width: 100%; }
    .stApp .stButton > button, .stApp .stDownloadButton > button {
        min-height: 3rem; padding: 0.5rem 1.4rem; border-radius: 10px;
        border: 2px solid %primary%; background: %surface%; color: %primary%;
        font-weight: 700;
    }
    .stApp .stButton > button[kind="primary"], .stApp .stDownloadButton > button[kind="primary"] {
        background: %primary%; color: %on_primary%;
    }
    .stApp .stButton > button:hover, .stApp .stDownloadButton > button:hover {
        background: %soft%; color: %ink%; border-color: %ink%;
    }
    .stApp .stButton > button[kind="primary"]:hover { background: %ink%; color: %surface%; }

    /* Tabs */
    .stApp [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 2px solid %border%; }
    .stApp [data-baseweb="tab"] {
        min-height: 3rem; padding: 0.4rem 1.1rem; font-weight: 700; color: %muted%;
        border-radius: 10px 10px 0 0;
    }
    .stApp [data-baseweb="tab"][aria-selected="true"] { color: %primary%; background: %soft%; }
    .stApp [data-baseweb="tab-highlight"] { background: %primary% !important; height: 4px !important; }

    /* Strong, always-visible keyboard focus */
    .stApp *:focus-visible, .stApp button:focus-visible, .stApp textarea:focus-visible,
    .stApp [data-baseweb="select"] *:focus-visible, .stApp [data-baseweb="tab"]:focus-visible {
        outline: 4px solid %focus% !important; outline-offset: 2px !important;
        box-shadow: none !important;
    }

    /* Panels and cards */
    .panel { background: %surface%; border: 2px solid %border%; border-radius: 14px;
             padding: 1.1rem 1.3rem; margin: 0.6rem 0 1rem 0; }
    .panel h3 { margin-top: 0; }
    .hero { background: %primary%; color: %on_primary%; border-radius: 16px;
            padding: 1.6rem 1.8rem; margin-bottom: 1.2rem; }
    .hero h1, .hero p { color: %on_primary% !important; margin: 0; }
    .hero p { margin-top: 0.5rem; max-width: 60ch; }
    .hero { border: 2px solid %primary%; }

    .result { border-radius: 14px; padding: 1.2rem 1.4rem; margin: 1rem 0; border: 3px solid; }
    .result .head { display: flex; align-items: center; gap: 0.8rem; font-weight: 700;
                    font-size: 1.35rem; }
    .result .icon { font-size: 2rem; }
    .result p { margin: 0.5rem 0 0 0; color: inherit !important; }
    .r-real { background: %good_bg%; color: %good%; border-color: %good%; }
    .r-fake { background: %bad_bg%; color: %bad%; border-color: %bad%; }
    .r-unsure { background: %warn_bg%; color: %warn%; border-color: %warn%; }
    .meter { margin-top: 0.9rem; }
    .meter .bar { height: 1.1rem; border-radius: 999px; background: %surface%;
                  border: 2px solid currentColor; overflow: hidden; }
    .meter .fill { height: 100%; background: currentColor; }
    .meter .label { margin-top: 0.3rem; font-weight: 700; }

    .chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.4rem 0 0.8rem 0; padding: 0;
             list-style: none; }
    .chips li { background: %soft%; border: 2px solid %border%; border-radius: 999px;
                padding: 0.15rem 0.8rem; font-weight: 700; color: %ink%; }

    table.cm { border-collapse: collapse; width: 100%; background: %surface%; }
    table.cm th, table.cm td { border: 2px solid %border%; padding: 0.6rem 0.8rem; text-align: center; }
    table.cm th { background: %soft%; }
    [data-testid="stMetric"] { background: %surface%; border: 2px solid %border%;
                               border-radius: 12px; padding: 0.7rem 1rem; }
    [data-testid="stMetricValue"] { color: %primary% !important; font-weight: 700; }
    [data-testid="stExpander"] { border: 2px solid %border%; border-radius: 12px; background: %surface%; }
    [data-testid="stAlert"] { border: 2px solid %border%; border-radius: 12px; }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation: none !important; transition: none !important; }
    }
    """
    css = css.replace("%IMPORT%", imp).replace("%SIZE%", str(size_pct))
    css = css.replace("%FONT%", font_family).replace("%SPACING%", spacing)
    for key, val in p.items():
        css = css.replace(f"%{key}%", val)
    return f"<style>{css}</style>"


# ----------------------------------------------------------------------------
# Data + model
# ----------------------------------------------------------------------------
def read_table(name: str, data: bytes) -> pd.DataFrame:
    """Read a CSV, or the first CSV inside a ZIP."""
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
            if not csvs:
                raise ValueError("This zip file has no .csv file inside it.")
            with z.open(csvs[0]) as f:
                return pd.read_csv(f)
    return pd.read_csv(io.BytesIO(data))


def local_datasets():
    found = []
    for folder in {APP_DIR, Path.cwd()}:
        found += [p for p in folder.glob("*.zip")] + [p for p in folder.glob("*.csv")]
    return sorted(set(found))


def guess_columns(df: pd.DataFrame):
    cols = list(df.columns)
    text_col = "text_" if "text_" in cols else next(
        (c for c in cols if df[c].dtype == object and df[c].astype(str).str.len().mean() > 30), cols[0]
    )
    label_col = "label" if "label" in cols else next((c for c in cols if c != text_col), cols[-1])
    return text_col, label_col


@st.cache_resource(show_spinner=False)
def train(cache_key: str, _texts, _labels):
    """Same pipeline as the notebook: CountVectorizer -> 80/20 split -> MultinomialNB."""
    texts = pd.Series(_texts).fillna("").astype(str)
    y = pd.Series(_labels).astype(str)
    vec = CountVectorizer()
    X = vec.fit_transform(texts)
    stratify = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    model = MultinomialNB().fit(X_tr, y_tr)
    return dict(
        vec=vec, model=model, classes=[str(c) for c in model.classes_],
        y_test=list(y_te), y_pred=list(model.predict(X_te)),
        n_train=X_tr.shape[0], n_test=X_te.shape[0],
    )


def p_fake(bundle, fake_label, texts):
    X = bundle["vec"].transform([str(t) for t in texts])
    proba = bundle["model"].predict_proba(X)
    return proba[:, bundle["classes"].index(fake_label)]


def verdict(prob_fake):
    """Return (kind, confidence)."""
    conf = max(prob_fake, 1 - prob_fake)
    if conf < SURE_ENOUGH:
        return "unsure", conf
    return ("fake" if prob_fake >= 0.5 else "real"), conf


def explain(bundle, fake_label, text, top=5):
    """Words in the review that pushed the model toward 'fake' or 'genuine'."""
    X = bundle["vec"].transform([text]).tocsr()
    if X.nnz == 0:
        return [], []
    fi = bundle["classes"].index(fake_label)
    gi = 1 - fi
    lp = bundle["model"].feature_log_prob_
    score = (lp[fi, X.indices] - lp[gi, X.indices]) * X.data
    words = bundle["vec"].get_feature_names_out()[X.indices]
    order = np.argsort(score)
    fake_words = [words[i] for i in order[::-1] if score[i] > 0][:top]
    real_words = [words[i] for i in order if score[i] < 0][:top]
    return fake_words, real_words


def metrics_for(bundle, fake_label):
    genuine = next(c for c in bundle["classes"] if c != fake_label)
    yt = [c == fake_label for c in bundle["y_test"]]
    yp = [c == fake_label for c in bundle["y_pred"]]
    cm = confusion_matrix(yt, yp, labels=[False, True])
    return dict(
        genuine=genuine,
        accuracy=accuracy_score(yt, yp),
        precision=precision_score(yt, yp, zero_division=0),
        recall=recall_score(yt, yp, zero_division=0),
        f1=f1_score(yt, yp, zero_division=0),
        cm=cm,
    )


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
KIND_TEXT = {
    "real": ("✅", "Likely genuine", "This looks like it was written by a real person."),
    "fake": ("⚠️", "Likely fake", "This looks like it may have been generated by a computer."),
    "unsure": ("❓", "Not sure", "The tool can't tell. Treat this review with care and read it yourself."),
}


def result_card(kind, prob_fake, conf):
    icon, title, blurb = KIND_TEXT[kind]
    pct_fake = round(prob_fake * 100)
    st.markdown(
        f"""
        <div class="result r-{kind}" role="status" aria-live="polite">
          <div class="head"><span class="icon" aria-hidden="true">{icon}</span><span>{title}</span></div>
          <p>{blurb}</p>
          <div class="meter">
            <div class="bar" aria-hidden="true"><div class="fill" style="width:{pct_fake}%"></div></div>
            <div class="label">Chance this review is fake: {pct_fake}% &nbsp;|&nbsp; Chance it is genuine: {100 - pct_fake}%</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chip_list(words):
    items = "".join(f"<li>{html.escape(w)}</li>" for w in words)
    return f'<ul class="chips">{items}</ul>'


def speak_button(text, p):
    page = """
    <button id="b" style="min-height:48px;padding:8px 20px;border-radius:10px;cursor:pointer;
      border:2px solid %primary%;background:%surface%;color:%primary%;font-size:18px;font-weight:700;
      font-family:inherit" aria-label="Read the result aloud. Press again to stop.">🔊 Read result aloud</button>
    <script>
      const msg = %TEXT%;
      const b = document.getElementById('b');
      b.addEventListener('click', () => {
        const s = window.speechSynthesis;
        if (!s) { b.textContent = 'Reading aloud is not supported in this browser'; return; }
        if (s.speaking) { s.cancel(); return; }
        const u = new SpeechSynthesisUtterance(msg); u.rate = 0.95; s.speak(u);
      });
    </script>
    """
    for k in ("primary", "surface"):
        page = page.replace(f"%{k}%", p[k])
    components.html(page.replace("%TEXT%", json.dumps(text)), height=70)


def confusion_figure(m, p):
    cmap = LinearSegmentedColormap.from_list("cm", [p["surface"], p["primary"]])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    fig.patch.set_facecolor(p["surface"])
    ax.set_facecolor(p["surface"])
    cm = m["cm"]
    ax.imshow(cm, cmap=cmap)
    ax.set_xticks([0, 1], ["Genuine", "Fake"], color=p["ink"], fontsize=13)
    ax.set_yticks([0, 1], ["Genuine", "Fake"], color=p["ink"], fontsize=13)
    ax.set_xlabel("What the tool said", color=p["ink"], fontsize=13)
    ax.set_ylabel("What the review really is", color=p["ink"], fontsize=13)
    peak = cm.max() if cm.max() else 1
    for i in range(2):
        for j in range(2):
            dark_cell = cm[i, j] / peak > 0.5
            on_primary_dark = p["on_primary"] if dark_cell else p["ink"]
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=18,
                    fontweight="bold", color=on_primary_dark)
    for spine in ax.spines.values():
        spine.set_edgecolor(p["border"])
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Sidebar: accessibility controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Make it comfortable")
    st.caption("Change how the page looks. Your choices apply right away.")
    size_name = st.radio("Text size", list(TEXT_SIZES), index=0, horizontal=False)
    contrast = st.radio("Colours", list(PALETTES), index=0,
                        help="High contrast uses a black background with yellow and white text.")
    font_name = st.selectbox("Font", list(FONTS), index=0)
    extra_spacing = st.checkbox("Extra spacing between lines and letters", value=False)
    st.divider()

pal = PALETTES[contrast]
fam, imp = FONTS[font_name]
st.markdown(build_css(pal, TEXT_SIZES[size_name], fam, imp, extra_spacing), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>Review Checker</h1>
      <p>Paste a product review and find out if it was probably written by a person
      or generated by a computer.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Step 1: load data and train the model
# ----------------------------------------------------------------------------
bundle = st.session_state.get("bundle")

if bundle is None:
    st.markdown('<div class="panel"><h3>Step 1: Choose your review data</h3>'
                "The tool learns from a file of reviews that are already labelled "
                "(for example, the <b>archive.zip</b> from your notebook). "
                "It needs one column with the review text and one column with the label.</div>",
                unsafe_allow_html=True)

    locals_ = local_datasets()
    choice = None
    if locals_:
        names = [p.name for p in locals_]
        pick = st.selectbox("Files found next to this app", ["(none)"] + names)
        if pick != "(none)":
            choice = (pick, (locals_[names.index(pick)]).read_bytes())
    up = st.file_uploader("Or upload a .csv or .zip file", type=["csv", "zip"])
    if up is not None:
        choice = (up.name, up.getvalue())

    if choice is None:
        st.info("Choose or upload a file to begin.")
        st.stop()

    try:
        df = read_table(*choice)
    except Exception as exc:  # noqa: BLE001
        st.error(f"We could not read that file. {exc} Please check it is a CSV or a zip that contains a CSV.")
        st.stop()

    t_guess, l_guess = guess_columns(df)
    cols = list(df.columns)
    c1, c2 = st.columns(2)
    text_col = c1.selectbox("Column with the review text", cols, index=cols.index(t_guess))
    label_col = c2.selectbox("Column with the label", cols, index=cols.index(l_guess))

    labels = sorted(df[label_col].dropna().astype(str).unique())
    if len(labels) != 2:
        st.error(f"The label column must have exactly 2 different values. This one has {len(labels)}.")
        st.stop()
    default_fake = labels.index("CG") if "CG" in labels else len(labels) - 1
    fake_label = st.selectbox(
        "Which label means a FAKE review?", labels, index=default_fake,
        help="In the common fake-reviews dataset, CG means computer-generated and OR means original.",
    )

    st.caption(f"{len(df):,} reviews found.")
    if st.button("Train the checker", type="primary"):
        with st.spinner("Learning from the reviews. This can take a few seconds..."):
            sub = df.dropna(subset=[label_col])
            key = f"{choice[0]}|{len(sub)}|{text_col}|{label_col}"
            st.session_state["bundle"] = train(key, sub[text_col].tolist(), sub[label_col].tolist())
            st.session_state["fake_label"] = fake_label
            st.session_state["source"] = choice[0]
        st.rerun()
    st.stop()

fake_label = st.session_state["fake_label"]
m = metrics_for(bundle, fake_label)

with st.sidebar:
    st.subheader("Your checker")
    st.write(f"**Data:** {st.session_state['source']}")
    st.write(f"**Accuracy:** {m['accuracy']:.1%}")
    if st.button("Use different data"):
        for k in ("bundle", "fake_label", "source", "last"):
            st.session_state.pop(k, None)
        st.rerun()

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Check one review", "Check many reviews", "How accurate is it?"])

# ---- Tab 1 ------------------------------------------------------------------
with tab1:
    st.subheader("Check one review")
    st.write("Type or paste a review below, then choose **Check this review**.")

    st.write("**Not sure what to try? Pick an example:**")
    ex_cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(ex_cols, EXAMPLES.items()):
        col.button(name, key=f"ex_{name}",
                   on_click=lambda t=text: st.session_state.update(review_text=t, last=None))

    review = st.text_area(
        "Review text", key="review_text", height=170,
        placeholder="Example: The product is excellent and I am very happy with my purchase.",
        help="Tip: you can also dictate with your device's voice typing.",
    )
    words_n = len(review.split())
    st.caption(f"{words_n} word{'s' if words_n != 1 else ''}. Longer reviews give more reliable answers.")

    b1, b2 = st.columns([2, 1])
    go = b1.button("Check this review", type="primary")
    b2.button("Clear",
              on_click=lambda: st.session_state.update(review_text="", last=None))

    if go:
        if not review.strip():
            st.warning("Please type or paste a review first.")
        else:
            st.session_state["last"] = review

    last = st.session_state.get("last")
    if last:
        pf = float(p_fake(bundle, fake_label, [last])[0])
        kind, conf = verdict(pf)
        result_card(kind, pf, conf)

        fw, rw = explain(bundle, fake_label, last)
        if fw or rw:
            with st.expander("Which words influenced this result?", expanded=False):
                if fw:
                    st.write("Words that pointed toward **fake**:")
                    st.markdown(chip_list(fw), unsafe_allow_html=True)
                if rw:
                    st.write("Words that pointed toward **genuine**:")
                    st.markdown(chip_list(rw), unsafe_allow_html=True)
                st.caption("This is a simple word-counting model. It looks at which words appear, "
                           "not at what the review means.")
        speak_button(
            f"{KIND_TEXT[kind][1]}. {KIND_TEXT[kind][2]} "
            f"Chance this review is fake: {round(pf * 100)} percent.", pal)
        st.caption("This result is a guide, not proof. Please use your own judgement too.")

# ---- Tab 2 ------------------------------------------------------------------
with tab2:
    st.subheader("Check many reviews")
    st.write("Upload a CSV file with a column of reviews. You get a results table you can download.")
    batch = st.file_uploader("Choose a .csv file", type=["csv"], key="batch_up")
    if batch is not None:
        try:
            bdf = pd.read_csv(batch)
        except Exception as exc:  # noqa: BLE001
            st.error(f"We could not read that file. {exc}")
            st.stop()
        guess = "text_" if "text_" in bdf.columns else bdf.columns[0]
        col = st.selectbox("Which column has the review text?", list(bdf.columns),
                           index=list(bdf.columns).index(guess))
        if st.button("Check all reviews", type="primary"):
            texts = bdf[col].fillna("").astype(str)
            probs = p_fake(bundle, fake_label, texts)
            kinds = [verdict(x)[0] for x in probs]
            out = pd.DataFrame({
                "Review": texts,
                "Result": [KIND_TEXT[k][1] for k in kinds],
                "Chance fake (%)": (probs * 100).round(0).astype(int),
            })
            st.session_state["batch_out"] = out
        out = st.session_state.get("batch_out")
        if out is not None:
            counts = out["Result"].value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("Likely genuine", int(counts.get("Likely genuine", 0)))
            c2.metric("Likely fake", int(counts.get("Likely fake", 0)))
            c3.metric("Not sure", int(counts.get("Not sure", 0)))
            st.dataframe(out, hide_index=True, height=360)
            st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                               file_name="review_check_results.csv", mime="text/csv", type="primary")
    else:
        st.info("Upload a file to begin.")

# ---- Tab 3 ------------------------------------------------------------------
with tab3:
    st.subheader("How accurate is it?")
    total = int(m["cm"].sum())
    right = int(m["cm"][0, 0] + m["cm"][1, 1])
    st.markdown(
        f"The checker learned from **{bundle['n_train']:,}** reviews, then was tested on "
        f"**{bundle['n_test']:,}** reviews it had never seen. It was right **{right:,} times out of {total:,}** "
        f"(**{m['accuracy']:.1%}**)."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{m['accuracy']:.1%}", help="Share of all reviews it got right.")
    c2.metric("Precision", f"{m['precision']:.1%}",
              help="When it says a review is fake, how often is that correct?")
    c3.metric("Recall", f"{m['recall']:.1%}", help="Of all the fake reviews, how many did it catch?")

    st.markdown(
        f"- When the checker says **fake**, it is correct **{m['precision']:.0%}** of the time.\n"
        f"- It catches **{m['recall']:.0%}** of the fake reviews."
    )

    st.write("**Where it was right and wrong** (the confusion matrix):")
    st.pyplot(confusion_figure(m, pal))
    plt.close("all")

    cm = m["cm"]
    with st.expander("Show the same numbers as a table (best for screen readers)"):
        st.markdown(
            f"""
            <table class="cm">
              <caption>Test results: real label compared with the tool's answer</caption>
              <thead><tr><th scope="col"></th><th scope="col">Tool said genuine</th><th scope="col">Tool said fake</th></tr></thead>
              <tbody>
                <tr><th scope="row">Really genuine</th><td>{cm[0, 0]:,} correct</td><td>{cm[0, 1]:,} wrong</td></tr>
                <tr><th scope="row">Really fake</th><td>{cm[1, 0]:,} wrong</td><td>{cm[1, 1]:,} correct</td></tr>
              </tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )
