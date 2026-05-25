import os
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis · LSTM",
    page_icon="🎭",
    layout="wide",
)

st.title("🎭 Sentiment Analysis on Reviews")
st.markdown(
    "Classify airline tweets as **Positive**, **Neutral**, or **Negative** "
    "using a deep-learning LSTM model."
)

# ── Lazy imports (heavy; cached so they run once) ────────────────────────────
@st.cache_resource(show_spinner="Loading modules…")
def load_modules():
    from data_loader import load_data
    from preprocess import run_eda, preprocess, clean_text
    from model import build_model, train_and_evaluate
    return load_data, run_eda, preprocess, clean_text, build_model, train_and_evaluate

load_data, run_eda, preprocess, clean_text, build_model, train_and_evaluate = load_modules()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
epochs      = st.sidebar.slider("Max epochs",       3, 20, 10)
batch_size  = st.sidebar.slider("Batch size",       16, 128, 64, step=16)
max_words   = st.sidebar.slider("Vocabulary size",  2000, 20000, 10000, step=1000)
max_len     = st.sidebar.slider("Max sequence len", 20, 100, 50, step=5)
test_size   = st.sidebar.slider("Test split %",     10, 40, 20) / 100

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_data, tab_eda, tab_train, tab_predict = st.tabs(
    ["📂 Data", "📊 EDA", "🏋️ Train Model", "🔮 Predict"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("Dataset")

    @st.cache_data(show_spinner="Loading dataset…")
    def get_data():
        return load_data()

    df = get_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews",  f"{len(df):,}")
    col2.metric("Positive",       f"{(df.sentiment=='positive').sum():,}")
    col3.metric("Negative",       f"{(df.sentiment=='negative').sum():,}")

    st.dataframe(df.sample(min(200, len(df)), random_state=0).reset_index(drop=True),
                 use_container_width=True, height=340)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    os.makedirs("plots", exist_ok=True)

    @st.cache_data(show_spinner="Generating EDA plots…")
    def make_eda_plots(_df):
        import seaborn as sns

        # sentiment distribution
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(x="sentiment", data=_df, palette="Set2",
                      order=["positive", "neutral", "negative"], ax=ax1)
        ax1.set_title("Sentiment Distribution")
        plt.tight_layout()

        # review length
        _df = _df.copy()
        _df["length"] = _df["text"].apply(lambda x: len(str(x).split()))
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        for label, colour in zip(["positive", "neutral", "negative"],
                                  ["#2ecc71", "#3498db", "#e74c3c"]):
            subset = _df[_df.sentiment == label]["length"]
            ax2.hist(subset, bins=30, alpha=0.6, label=label, color=colour)
        ax2.set_title("Review Length by Sentiment")
        ax2.legend()
        plt.tight_layout()

        return fig1, fig2

    df = get_data()
    fig1, fig2 = make_eda_plots(df)

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(fig1)
    with c2:
        st.pyplot(fig2)

    # Pre-saved plot images (from original plots/ folder)
    if os.path.exists("plots"):
        saved = [f for f in os.listdir("plots") if f.endswith(".png")]
        if saved:
            st.markdown("---")
            st.markdown("**Pre-saved plots from last training run:**")
            cols = st.columns(len(saved))
            for col, fname in zip(cols, saved):
                col.image(f"plots/{fname}", caption=fname, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · TRAIN
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.subheader("Train the LSTM Model")

    # cache preprocessing separately so it isn't re-run on every widget change
    @st.cache_data(show_spinner="Preprocessing text…")
    def get_preprocessed(_df, _max_words, _max_len):
        return preprocess(_df, max_words=_max_words, max_len=_max_len)

    if st.button("🚀 Start Training", type="primary"):
        df = get_data()
        with st.spinner("Preprocessing…"):
            X, y, tokenizer, le = get_preprocessed(df, max_words, max_len)
        st.success(f"Preprocessed  — X shape: {X.shape}  |  classes: {list(le.classes_)}")

        # ── manual train/test split + fit ────────────────────────────────────
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        from tensorflow.keras.callbacks import EarlyStopping
        import seaborn as sns

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

        num_classes = len(le.classes_)
        model = build_model(vocab_size=max_words, max_len=max_len, num_classes=num_classes)

        progress_bar = st.progress(0, text="Training…")
        epoch_log    = st.empty()

        class StreamlitCallback:
            def __init__(self, total):
                self.total = total
                self.logs  = []

            def on_epoch_end(self, epoch, logs=None):
                pct = int((epoch + 1) / self.total * 100)
                progress_bar.progress(pct, text=f"Epoch {epoch+1}/{self.total}")
                self.logs.append(logs or {})
                epoch_log.write(
                    f"**Epoch {epoch+1}** — "
                    f"loss: {logs.get('loss',0):.4f}  "
                    f"acc: {logs.get('accuracy',0):.4f}  "
                    f"val_loss: {logs.get('val_loss',0):.4f}  "
                    f"val_acc: {logs.get('val_accuracy',0):.4f}"
                )

        cb = StreamlitCallback(epochs)

        from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping
        keras_cb = LambdaCallback(on_epoch_end=lambda e, logs: cb.on_epoch_end(e, logs))
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            epochs=epochs, batch_size=batch_size,
            validation_split=0.1,
            callbacks=[keras_cb, es],
            verbose=0,
        )
        progress_bar.progress(100, text="Done!")

        # ── metrics ──────────────────────────────────────────────────────────
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        st.metric("Test Accuracy", f"{acc:.2%}")

        y_pred = np.argmax(model.predict(X_test), axis=1)
        report = classification_report(
            y_test, y_pred,
            target_names=list(le.classes_),
            output_dict=True,
        )
        import pandas as pd
        st.dataframe(pd.DataFrame(report).T.style.format("{:.2f}"),
                     use_container_width=True)

        # ── training curves ───────────────────────────────────────────────────
        h = history.history
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(h["accuracy"],     label="train")
        axes[0].plot(h["val_accuracy"], label="val")
        axes[0].set_title("Accuracy")
        axes[0].legend()
        axes[1].plot(h["loss"],     label="train")
        axes[1].plot(h["val_loss"], label="val")
        axes[1].set_title("Loss")
        axes[1].legend()
        plt.tight_layout()
        st.pyplot(fig)

        # ── confusion matrix ─────────────────────────────────────────────────
        cm = confusion_matrix(y_test, y_pred)
        fig2, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig2)

        # persist model + artefacts for the Predict tab
        st.session_state["model"]     = model
        st.session_state["tokenizer"] = tokenizer
        st.session_state["le"]        = le
        st.session_state["max_len"]   = max_len
        st.success("Model saved — switch to the **Predict** tab to try it live!")

    else:
        st.info("Click **Start Training** to build and evaluate the LSTM model.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("Live Sentiment Prediction")

    EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😠"}
    COLOR = {"positive": "green", "neutral": "orange", "negative": "red"}

    if "model" not in st.session_state:
        st.warning("Train the model first (go to the **Train Model** tab).")
    else:
        model     = st.session_state["model"]
        tokenizer = st.session_state["tokenizer"]
        le        = st.session_state["le"]
        stored_len = st.session_state["max_len"]

        user_input = st.text_area(
            "Enter a review or tweet:",
            placeholder="e.g.  The flight was delayed and no one helped us at all!",
            height=100,
        )

        if st.button("Predict", type="primary"):
            if not user_input.strip():
                st.warning("Please enter some text.")
            else:
                from tensorflow.keras.preprocessing.sequence import pad_sequences

                cleaned  = clean_text(user_input)
                seq      = tokenizer.texts_to_sequences([cleaned])
                padded   = pad_sequences(seq, maxlen=stored_len, padding="post")
                probs    = model.predict(padded, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                label    = le.inverse_transform([pred_idx])[0]
                emoji    = EMOJI.get(label, "")
                colour   = COLOR.get(label, "blue")

                st.markdown(
                    f"### Prediction: :{colour}[{label.upper()} {emoji}]"
                )

                st.markdown("**Confidence scores:**")
                for cls, prob in zip(le.classes_, probs):
                    st.progress(float(prob), text=f"{cls}: {prob:.1%}")

        st.markdown("---")
        st.markdown("**Try these examples:**")
        examples = [
            "Amazing service, crew was super friendly and flight was on time!",
            "Flight delayed 3 hours, no updates, terrible experience.",
            "Flight was okay, nothing special but got there safely.",
        ]
        for ex in examples:
            if st.button(ex, key=ex):
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                cleaned  = clean_text(ex)
                seq      = tokenizer.texts_to_sequences([cleaned])
                padded   = pad_sequences(seq, maxlen=stored_len, padding="post")
                probs    = model.predict(padded, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                label    = le.inverse_transform([pred_idx])[0]
                emoji    = EMOJI.get(label, "")
                colour   = COLOR.get(label, "blue")
                st.markdown(
                    f"**\"{ex}\"** → :{colour}[{label.upper()} {emoji}]  "
                    f"*(confidence: {probs[pred_idx]:.1%})*"
                )