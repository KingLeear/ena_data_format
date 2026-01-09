import streamlit as st
import pandas as pd
from pathlib import Path
import yaml

from ena_tool import (
    segment_csv,
    step_paradigms,
    step_train_multiclass,
    step_predict_multiclass,
    step_binarize_ena,
)

st.set_page_config(page_title="ENA Pipeline", layout="wide")
st.title("ENA End-to-End Pipeline")

# ----------------------------
# OpenAI API Key input
# ----------------------------
st.sidebar.header("Input OpenAI API Key")

openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API key",
    type="password",
    help="Your key is used only for this session and not stored."
)

if openai_api_key:
    import os
    os.environ["OPENAI_API_KEY"] = openai_api_key
    st.sidebar.success("API key loaded for this session.")
else:
    st.sidebar.warning("No API key set. Paradigm generation will not work.")

# ----------------------------
# 0) Session state for concepts
# ----------------------------
if "concepts" not in st.session_state:
    st.session_state["concepts"] = [
        {"code": "C1", "label": "Concept 1", "definition": ""},
        {"code": "C2", "label": "Concept 2", "definition": ""},
    ]

st.write("Define concept labels → generate paradigms → train model → upload raw data → run full ENA pipeline.")

st.divider()

# ----------------------------
# 1) Concept / Label editor
# ----------------------------
# st.header("1.Define concept labels (multiclass)")

# with st.expander("Edit concepts", expanded=True):
#     cols_header = st.columns([1, 2, 3, 1])
#     cols_header[0].markdown("**Code**")
#     cols_header[1].markdown("**Label**")
#     cols_header[2].markdown("**Definition**")
#     cols_header[3].markdown("**Remove**")

#     for i, c in enumerate(st.session_state["concepts"]):
#         cols = st.columns([1, 2, 3, 1])
#         c["code"] = cols[0].text_input("", value=c.get("code", ""), key=f"code_{i}")
#         c["label"] = cols[1].text_input("", value=c.get("label", ""), key=f"label_{i}")
#         c["definition"] = cols[2].text_area("", value=c.get("definition", ""), key=f"def_{i}", height=80)
#         if cols[3].button("Remove", key=f"del_{i}"):
#             st.session_state["concepts"].pop(i)
#             st.rerun()

#     c1, c2 = st.columns([1, 3])
#     if c1.button("Add concept"):
#         st.session_state["concepts"].append({"code": "", "label": "", "definition": ""})
#         st.rerun()

# # Validate concepts
# def _validate_concepts(concepts: list[dict]) -> list[str]:
#     errs = []
#     codes = [c.get("code", "").strip() for c in concepts]
#     if any(not x for x in codes):
#         errs.append("Some concept codes are empty.")
#     if len(set(codes)) != len(codes):
#         errs.append("Duplicate concept codes detected.")
#     return errs

# st.divider()
st.header("1. Define concept labels (multiclass)")

# init
if "concepts" not in st.session_state:
    st.session_state["concepts"] = []

st.caption("Upload a CSV with concept definitions. Supported columns: "
           "`code,label,definition` (recommended) or `concept,definition` (2-column).")

concepts_file = st.file_uploader("Upload concepts CSV", type=["csv"], key="concepts_csv")

cA, cB = st.columns([1, 1])
with cA:
    st.download_button(
        "Download template CSV",
        data=("code,label,definition\n"
              "C1,Concept 1,Define concept 1 here\n"
              "C2,Concept 2,Define concept 2 here\n").encode("utf-8-sig"),
        file_name="concepts_template.csv",
        mime="text/csv",
    )
with cB:
    if st.button("Clear loaded concepts"):
        st.session_state["concepts"] = []
        st.rerun()

def _normalize_concepts_df(df: pd.DataFrame) -> list[dict]:
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols

    # Option A: code/label/definition
    if {"code", "label", "definition"}.issubset(set(df.columns)):
        out = []
        for _, r in df.iterrows():
            out.append({
                "code": str(r["code"]).strip(),
                "label": str(r["label"]).strip(),
                "definition": "" if pd.isna(r["definition"]) else str(r["definition"]).strip(),
            })
        return out

    # Option B: concept/definition (2 columns)
    if {"concept", "definition"}.issubset(set(df.columns)):
        out = []
        for _, r in df.iterrows():
            c = str(r["concept"]).strip()
            out.append({
                "code": c,          
                "label": c,         
                "definition": "" if pd.isna(r["definition"]) else str(r["definition"]).strip(),
            })
        return out

    raise ValueError(f"CSV columns not recognized. Got: {list(df.columns)}. "
                     "Need either code,label,definition OR concept,definition.")

def _validate_concepts(concepts: list[dict]) -> list[str]:
    errs = []
    codes = [c.get("code", "").strip() for c in concepts]
    if len(concepts) == 0:
        errs.append("No concepts loaded.")
        return errs
    if any(not x for x in codes):
        errs.append("Some concept codes are empty.")
    if len(set(codes)) != len(codes):
        errs.append("Duplicate concept codes detected.")
    return errs

# Load CSV
if concepts_file is not None:
    try:
        df_c = pd.read_csv(concepts_file)
        concepts = _normalize_concepts_df(df_c)
        st.session_state["concepts"] = concepts
        st.success(f"Loaded {len(concepts)} concepts from CSV.")
        st.dataframe(df_c.head(20))
    except Exception as e:
        st.error(f"Failed to load concepts CSV: {e}")

# Preview + optional manual edit
errs = _validate_concepts(st.session_state["concepts"])
if errs:
    st.warning(" | ".join(errs))
else:
    st.write("Current concepts:")
    st.dataframe(pd.DataFrame(st.session_state["concepts"]))

with st.expander("Optional: manually tweak concepts", expanded=False):
    if len(st.session_state["concepts"]) == 0:
        st.info("Upload a concepts CSV first.")
    else:
        for i, c in enumerate(st.session_state["concepts"]):
            cols = st.columns([1, 2, 3])
            c["code"] = cols[0].text_input("code", value=c.get("code", ""), key=f"code_{i}")
            c["label"] = cols[1].text_input("label", value=c.get("label", ""), key=f"label_{i}")
            c["definition"] = cols[2].text_area("definition", value=c.get("definition", ""), key=f"def_{i}", height=80)

# ----------------------------
# 2) Generate paradigms + train model
# ----------------------------
st.header("2. Generate paradigms and train model")


train_col1, train_col2, train_col3 = st.columns([1, 1, 1])
model_dir = train_col1.text_input("Model output directory", value="model_out")
paradigm_lang = train_col2.selectbox("Paradigm language", ["zh", "en"], index=0)
paradigm_n = train_col3.number_input("Paradigms per concept", min_value=5, max_value=200, value=50, step=5)

train_col4, train_col5, train_col6 = st.columns([1, 1, 1])
backbone = train_col4.selectbox("Backbone", ["bert", "roberta"], index=0)
epochs = train_col5.number_input("Epochs", min_value=1, max_value=10, value=3, step=1)
batch_size = train_col6.number_input("Batch size", min_value=2, max_value=64, value=8, step=2)

openai_model = st.text_input("OpenAI model (for paradigms)", value="gpt-5.2")

out_dir = Path(model_dir)
out_dir.mkdir(parents=True, exist_ok=True)

if st.button("Generate paradigms + Train model"):
    st.write(">>> TRAIN BUTTON CLICKED")

    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        errs = _validate_concepts(st.session_state["concepts"])
        if errs:
            st.error(" | ".join(errs))
        else:
            try:
                concepts_yaml = Path("data/_tmp_concepts.yaml")
                paradigms_csv = Path("data/_tmp_paradigms.csv")
                concepts_yaml.parent.mkdir(parents=True, exist_ok=True)
                paradigms_csv.parent.mkdir(parents=True, exist_ok=True)

                concepts_payload = {"concepts": st.session_state["concepts"]}
                concepts_yaml.write_text(
                    yaml.safe_dump(concepts_payload, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )

                with st.spinner("Generating paradigms via OpenAI..."):
                    step_paradigms(
                        concepts_yaml=concepts_yaml,
                        out_csv=paradigms_csv,
                        lang=paradigm_lang,
                        n=int(paradigm_n),
                        model=openai_model,
                    )

                progress = st.progress(0.0)
                with st.spinner("Training multiclass classifier..."):
                    metrics = step_train_multiclass(
                        paradigms_csv=paradigms_csv,
                        out_dir=out_dir,
                        backbone=backbone,
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        max_len=128,
                        seed=42,
                        progress_bar=progress,
                    )

                
                st.write(">>> AFTER TRAIN")
                st.session_state["train_metrics"] = metrics
                st.success("Model trained successfully.")
                st.write("Training metrics:")
                metrics_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
                st.table(metrics_df)
                

            except Exception as e:
                st.error(f"Training failed: {e}")

st.divider()

# ----------------------------
# 3) Upload raw data + run full pipeline
# ----------------------------

st.header("3. Upload raw data and run full pipeline")

raw_file = st.file_uploader("Upload raw CSV (student_id + text)", type=["csv"])

if raw_file is not None:
    raw_df = pd.read_csv(raw_file)
    st.subheader("Raw data preview")
    st.dataframe(raw_df.head())

    text_col = st.selectbox("Text column", raw_df.columns)
    id_cols = st.multiselect("ID columns to keep", raw_df.columns, default=[raw_df.columns[0]])

    group_choice = st.selectbox("Group by (segment numbering)", ["(none)"] + list(raw_df.columns))
    group_col = None if group_choice == "(none)" else group_choice

    seg_lang = st.selectbox("Segmentation language", ["auto", "zh", "en"], index=0)

    st.subheader("Binarize settings")
    mode = st.selectbox("Binarization mode", ["top1", "threshold"], index=0)
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01) if mode == "threshold" else 0.5

    if st.button("Run full pipeline 🚀"):
        import os

        tmp_raw   = Path("data/_tmp_raw.csv")
        tmp_units = Path("data/_tmp_units.csv")
        tmp_pred  = Path("data/_tmp_pred.csv")
        tmp_ena   = Path("data/_tmp_ena.csv")

        # ===== Params =====
        min_len_zh = 2
        min_len_en = 1
        pred_text_col = "text"
        prob_prefix = "p_"

        st.write("CWD =", os.getcwd())
        st.write("Paths:", str(tmp_raw.resolve()), str(tmp_units.resolve()), str(tmp_pred.resolve()), str(tmp_ena.resolve()))

        # ===== Step 1: Write raw =====
        tmp_raw.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(tmp_raw, index=False, encoding="utf-8-sig")
        st.success(f"Step 1 OK: wrote raw -> {tmp_raw} ({len(raw_df)} rows)")

        if text_col not in raw_df.columns:
            st.error(f"text_col '{text_col}' not found in raw_df columns: {list(raw_df.columns)}")
            st.stop()

        if raw_df[text_col].fillna("").astype(str).str.strip().eq("").all():
            st.error(f"text_col '{text_col}' is empty for all rows.")
            st.stop()

        # ===== Step 2: Segmentation =====
        with st.spinner("Step 2: Segmenting text..."):
            try:
                segment_csv(
                    in_csv=tmp_raw,
                    out_csv=tmp_units,
                    text_col=text_col,
                    id_cols=id_cols,
                    group_col=group_col,
                    lang=seg_lang,
                    min_len_zh=min_len_zh,
                    min_len_en=min_len_en,
                )
            except Exception as e:
                st.error(f"Segmentation failed: {e}")
                st.stop()

        if not tmp_units.exists():
            st.error("Segmentation finished but tmp_units was not created.")
            st.stop()

        df_units = pd.read_csv(tmp_units)
        st.write("Step 2 output columns:", list(df_units.columns))
        st.write("Step 2 units rows:", len(df_units))
        st.dataframe(df_units.head(10))

        if len(df_units) == 0:
            st.warning("No segmented units produced. Check lang/min_len/text_col settings.")
            st.stop()

        if pred_text_col not in df_units.columns:
            st.error(f"Units missing '{pred_text_col}' column. Existing: {list(df_units.columns)}")
            st.stop()

        # ===== Step 3: Prediction =====
        with st.spinner("Step 3: Predicting concepts..."):
            try:
                step_predict_multiclass(
                    units_csv=tmp_units,
                    model_dir=Path(model_dir),
                    out_csv=tmp_pred,
                    text_col=pred_text_col,
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        if not tmp_pred.exists():
            st.error("Prediction finished but tmp_pred was not created.")
            st.stop()

        df_pred = pd.read_csv(tmp_pred)
        st.write("Step 3 pred rows:", len(df_pred))
        st.dataframe(df_pred.head(10))

        prob_cols = [c for c in df_pred.columns if c.startswith(prob_prefix)]
        if len(prob_cols) == 0:
            st.error(f"No probability columns found with prefix '{prob_prefix}'.")
            st.stop()

        # ===== Step 4: Binarize for ENA =====
        with st.spinner("Step 4: Binarizing for ENA..."):
            try:
                step_binarize_ena(
                    in_csv=tmp_pred,
                    out_csv=tmp_ena,
                    keep_cols=[c for c in ["student_id", "__row_id", pred_text_col] if c in df_pred.columns],
                    mode=mode,                  
                    threshold=float(threshold),
                    prob_prefix=prob_prefix,
                )
            except Exception as e:
                st.error(f"ENA binarization failed: {e}")
                st.stop()

        if not tmp_ena.exists():
            st.error("ENA binarization finished but tmp_ena was not created.")
            st.stop()

        ena_df = pd.read_csv(tmp_ena)
        st.success(f"Step 4 OK: wrote ENA-coded -> {tmp_ena} ({len(ena_df)} rows)")
        st.dataframe(ena_df.head(10))

        st.download_button(
            "Download ENA CSV",
            data=ena_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="ena_output.csv",
            mime="text/csv",
        )