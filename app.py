import streamlit as st
import pandas as pd
from pathlib import Path
import yaml
import json, subprocess
from pathlib import Path
import subprocess
import streamlit as st

from ena_tool import (
    segment_csv,
    step_paradigms,
    step_train_multiclass,
    step_predict_multiclass,
    step_binarize_ena,
)

st.set_page_config(page_title="ENA Pipeline", layout="wide")
st.title("ENA End-to-End Pipeline")

# OpenAI API Key input
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

# 0) Session state for concepts

if "concepts" not in st.session_state:
    st.session_state["concepts"] = [
        {"code": "C1", "label": "Concept 1", "definition": ""},
        {"code": "C2", "label": "Concept 2", "definition": ""},
    ]

st.write("Define concept labels → generate paradigms → train model → upload raw data → run full ENA pipeline.")

st.divider()


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


# 2) Generate paradigms + train model


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

# 3) Upload raw data + run full pipeline

#Init session state
if "pred_ready" not in st.session_state:
    st.session_state.pred_ready = False
if "ena_ready" not in st.session_state:
    st.session_state.ena_ready = False

if "tmp_pred" not in st.session_state:
    st.session_state.tmp_pred = None
if "tmp_ena" not in st.session_state:
    st.session_state.tmp_ena = None
if "prob_prefix" not in st.session_state:
    st.session_state.prob_prefix = "p_"

if "mode" not in st.session_state:
    st.session_state.mode = "top1"
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5

############################################################################

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

    #  Pre-seg filter UI (raw-level, immediate) 
    st.subheader("Raw text audit (before pipeline)")
    enable_pre_filter = st.checkbox("Filter short raw texts before segmentation", value=True)
    min_chars_raw = st.number_input("Min chars (raw text)", min_value=0, value=20, step=1)

    # Raw audit + hold-out (NO pipeline needed)
    raw_df["_raw_text"] = raw_df[text_col].fillna("").astype(str).str.strip()
    raw_df["_raw_char_len"] = raw_df["_raw_text"].str.len()

    if enable_pre_filter and min_chars_raw > 0:
        short_raw_df = raw_df[raw_df["_raw_char_len"] < int(min_chars_raw)].copy()
        kept_raw_df  = raw_df[raw_df["_raw_char_len"] >= int(min_chars_raw)].copy()
    else:
        short_raw_df = raw_df.iloc[0:0].copy()
        kept_raw_df  = raw_df.copy()

    st.caption(f"Kept rows: {len(kept_raw_df)} | Held-out short rows: {len(short_raw_df)}")

    with st.expander(f"Preview held-out short rows (< {int(min_chars_raw)} chars)", expanded=True):
        show_cols = [c for c in [*id_cols, group_col, text_col, "_raw_char_len"] if c and c in short_raw_df.columns]
        if not show_cols:
            show_cols = [text_col, "_raw_char_len"]
        st.dataframe(short_raw_df[show_cols].head(50))

        st.download_button(
            "Download short raw CSV",
            data=short_raw_df.drop(columns=["_raw_text"], errors="ignore").to_csv(index=False).encode("utf-8-sig"),
            file_name="short_raw.csv",
            mime="text/csv",
        )

    # Store settings for Step 4 and pipeline
    st.session_state["mode"] = mode
    st.session_state["threshold"] = float(threshold)
    st.session_state["raw_df_for_pipeline"] = kept_raw_df.drop(columns=["_raw_text"], errors="ignore")
    st.session_state["short_raw_df"] = short_raw_df.drop(columns=["_raw_text"], errors="ignore")
    st.session_state["enable_pre_filter"] = bool(enable_pre_filter)
    st.session_state["min_chars_raw"] = int(min_chars_raw)

if raw_file is not None:
    
    if st.button("Run full pipeline"):
        import os
        from pathlib import Path

        tmp_raw   = Path("data/_tmp_raw.csv")
        tmp_units = Path("data/_tmp_units.csv")
        tmp_pred  = Path("data/_tmp_pred.csv")
        tmp_ena   = Path("data/_tmp_ena.csv")

        # Params
        min_len_zh = 2
        min_len_en = 1
        pred_text_col = "text"
        prob_prefix = "p_"

        for p in [tmp_units, tmp_pred, tmp_ena]:
            if p.exists():
                p.unlink()

        st.write("CWD =", os.getcwd())
        st.write("Paths:", str(tmp_raw.resolve()), str(tmp_units.resolve()), str(tmp_pred.resolve()), str(tmp_ena.resolve()))

        raw_df_for_pipeline = st.session_state.get("raw_df_for_pipeline", raw_df)

        tmp_raw.parent.mkdir(parents=True, exist_ok=True)
        raw_df_for_pipeline.to_csv(tmp_raw, index=False, encoding="utf-8-sig")
        st.success(f"Step 1 OK: wrote raw -> {tmp_raw} ({len(raw_df_for_pipeline)} rows)")

        if text_col not in raw_df_for_pipeline.columns:
            st.error(f"text_col '{text_col}' not found in columns: {list(raw_df_for_pipeline.columns)}")
            st.stop()

        if raw_df_for_pipeline[text_col].fillna("").astype(str).str.strip().eq("").all():
            st.error(f"text_col '{text_col}' is empty for all rows after filtering.")
            st.stop()

        # Step 2: Segmentation 

        with st.spinner("Step 2: Segmenting text..."):
            try:
                short_raw_path = Path("data/_tmp_short_raw.csv")
                pre_min = int(st.session_state.get("min_chars_raw", 0))
                pre_on  = bool(st.session_state.get("enable_pre_filter", False))    
                segment_csv(
                    in_csv=tmp_raw,
                    out_csv=tmp_units,
                    text_col=text_col,
                    id_cols=id_cols,
                    group_col=group_col,
                    lang=seg_lang,
                    min_len_zh=min_len_zh,
                    min_len_en=min_len_en,
                    keep_cols=None,
                    pre_filter_min_chars=pre_min if pre_on else 0,
                    pre_filter_out_csv=short_raw_path if pre_on else None,
                )
            except Exception as e:
                st.error(f"Segmentation failed: {e}")
                st.stop()


        if enable_pre_filter and short_raw_path.exists():
            short_df = pd.read_csv(short_raw_path)
            st.subheader(f"Short raw texts (< {int(min_chars_raw)} chars)")
            st.write("Rows:", len(short_df))
            st.dataframe(short_df.head(50))  
            



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

            # Step 3: Prediction 
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


        # Step 3 done
        st.session_state.pred_ready = True
        st.session_state.tmp_pred = str(tmp_pred)
        st.session_state.tmp_ena  = str(tmp_ena)
        st.session_state.prob_prefix = prob_prefix
        st.success("Step 3 OK. Now select columns for Step 4 below.")

        

# Step 4 UI & Run (outside the "Run full pipeline" button) 
if st.session_state.pred_ready and st.session_state.tmp_pred:
    tmp_pred = Path(st.session_state.tmp_pred)
    tmp_ena  = Path(st.session_state.tmp_ena)
    prob_prefix = st.session_state.prob_prefix

    df_pred = pd.read_csv(tmp_pred)

    st.subheader("Step 4: Binarize for ENA")
    prob_cols = [c for c in df_pred.columns if c.startswith(prob_prefix)]
    candidate_keep = [c for c in df_pred.columns if c not in prob_cols]

    with st.form("ena_keepcols_form"):
        keep_cols = st.multiselect(
            "Keep columns in _tmp_ena.csv",
            options=candidate_keep,
            default=[],
            key="step4_keep_cols",
        )
        run_step4 = st.form_submit_button("Run Step 4: Binarize for ENA")

    if run_step4:
        if not keep_cols:
            st.error("Please select at least one column to keep.")
            st.stop()

        with st.spinner("Step 4: Binarizing for ENA..."):
            step_binarize_ena(
                in_csv=tmp_pred,
                out_csv=tmp_ena,
                keep_cols=keep_cols,
                mode=st.session_state.mode,
                threshold=float(st.session_state.threshold),
                prob_prefix=prob_prefix,
            )

        if not tmp_ena.exists():
            st.error("ENA binarization finished but tmp_ena was not created.")
            st.stop()

        st.session_state.ena_ready = True
        st.session_state.tmp_ena = str(tmp_ena)

        ena_df = pd.read_csv(tmp_ena)
        st.success(f"Step 4 OK: wrote ENA-coded -> {tmp_ena} ({len(ena_df)} rows)")
        st.dataframe(ena_df.head(10))

        st.download_button(
            "Download ENA CSV",
            data=ena_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="ena_output.csv",
            mime="text/csv",
            key="download_ena_csv",
        )


# Step 5: Build ENA set (.RData)


OUT_DIR = Path("/Users/tiffanyhsu/Desktop/ENA/ena-tool/ena_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if st.session_state.get("ena_ready") and st.session_state.get("tmp_ena"):
    tmp_ena_path = Path(st.session_state.tmp_ena)

    if not tmp_ena_path.exists():
        st.warning("ENA-coded CSV not found. Please rerun Step 4.")
        st.session_state.ena_ready = False
        st.stop()

    ena_df = pd.read_csv(tmp_ena_path)  
    st.subheader("Step 5: Build ENA set (R)")

    all_cols = list(ena_df.columns)

    unitCols = st.multiselect(
    "unitCols (units)",
    options=all_cols,
    default=[],
    key="step5_unitCols",
    help="Choose columns that uniquely identify a unit (e.g., UserID, Activity...)."
    )

    conversationCols = st.multiselect(
    "conversationCols",
    options=all_cols,
    default=[],
    key="step5_conversationCols",
    )

    window_size_back = st.number_input(
    "window.size.back (co-occurrence window)",
    min_value=1,
    max_value=20,
    value=2,
    step=1,
    key="step5_window"
    )

    groupsVar = st.selectbox(
        "groupsVar",
        options=["(none)"] + all_cols,
        key="step5_groupsVar",
    )

    if groupsVar != "(none)" and groupsVar in ena_df.columns:
        groups_options = sorted(ena_df[groupsVar].dropna().astype(str).unique().tolist())
    else:
        groups_options = []

    groups = st.multiselect(
        "groups (levels of groupsVar)",
        options=groups_options,
        key="step5_groups",
    )

    codesExclude = st.multiselect(
    "Exclude columns from ENA codes (non-concept columns)",
    options=all_cols,
    default=[],
    key="step5_codesExclude",
    help="Select columns that should NOT be treated as ENA concept nodes."
    )   

    if len(codesExclude) == 0:
        st.warning(
            "You have not excluded any columns. "
            "This means ALL columns will be treated as ENA codes. "
            "Please confirm this is intentional."
        )


    cfg = {
        "unitCols": unitCols,
        "conversationCols": conversationCols,
        "groupsVar": None if groupsVar == "(none)" else groupsVar,
        "groups": groups,
        "codesExclude": codesExclude,
        "model": "EndPoint",
        "window.size.back": int(window_size_back),
        "weight.by": "$",
        "object_name": "set.ena",
    }

    if st.button("Run Step 5: Build ENA set (.RData)", key="btn_step5_run"):
        ena_csv_path = OUT_DIR / "ena_output_latest.csv"
        ena_cfg_path = OUT_DIR / "ena_config_latest.json"
        ena_rdata_path = OUT_DIR / "ena_set_latest.RData"

        ena_df.to_csv(ena_csv_path, index=False, encoding="utf-8-sig")
        ena_cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


        run_ena_path = Path(__file__).parent / "run_ena.R"

        cmd = ["Rscript", str(run_ena_path), str(ena_csv_path), str(ena_rdata_path), str(ena_cfg_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)

        if res.returncode != 0:
            st.error("Rscript failed")
            st.code(res.stderr)
        else:
            st.success("ENA set built successfully")
            if res.stdout.strip():
                st.code(res.stdout)

            with open(ena_rdata_path, "rb") as f:
                st.download_button(
                    "Download ENA set (.RData)",
                    data=f,
                    file_name=ena_rdata_path.name,
                    mime="application/octet-stream",
                    key="dl_step5_rdata",
                )
else:
    st.info("Run Step 4 first. After ENA-coded CSV is generated, Step 5 will appear here.")



st.subheader("Step 6: Launch ENA3d (Shiny folder)")

ena3d_dir = st.text_input(
    "ENA3d folder path (must contain app.R)",
    value="/Users/tiffanyhsu/Desktop/ENA/ena-tool/ENA_3D/R",
    key="ena3d_dir",
)

port = st.number_input(
    "Port",
    min_value=1024,
    max_value=65535,
    value=3838,
    step=1,
    key="ena3d_port",
)



app_r = Path(ena3d_dir) / "app.R"
if not app_r.exists():
    st.error(f"Cannot find app.R：{app_r}\nPlease check the ENA3d folder path.")
    st.stop()

launch_r = Path(ena3d_dir) / "launch_shiny_app.R"

if "ena3d_proc" not in st.session_state:
    st.session_state.ena3d_proc = None
if "ena3d_last_stdout" not in st.session_state:
    st.session_state.ena3d_last_stdout = ""
if "ena3d_last_stderr" not in st.session_state:
    st.session_state.ena3d_last_stderr = ""

col1, col2, col3 = st.columns([1, 1, 2])

def proc_is_running(p):
    return (p is not None) and (p.poll() is None)

with col1:
    if st.button("Launch ENA3d", key="btn_launch_ena3d"):
        if not Path(ena3d_dir).exists():
            st.error(f"Folder not found: {ena3d_dir}")
        elif not launch_r.exists():
            st.error(f"launch_shiny_app.R not found: {launch_r}")
        else:
            proc = st.session_state.ena3d_proc
            if proc_is_running(proc):
                st.warning("ENA3d is already running.")
            else:
                cmd = ["Rscript", str(launch_r), ena3d_dir, str(int(port))]
                st.session_state.ena3d_proc = subprocess.Popen(
                    cmd,
                    cwd=ena3d_dir,  
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                st.success("ENA3d launched.")

with col2:
    if st.button("Stop ENA3d", key="btn_stop_ena3d"):
        proc = st.session_state.ena3d_proc
        if proc_is_running(proc):
            proc.terminate()
            st.success("ENA3d stopped.")
        else:
            st.info("ENA3d is not running.")

with col3:
    proc = st.session_state.ena3d_proc
    if proc_is_running(proc):
        st.markdown("**Status:** Running")
    else:
        st.markdown("**Status:** Stopped")

st.write(f"Open ENA3d: http://127.0.0.1:{int(port)}")

# Show logs (help you debug immediately)
with st.expander("ENA3d logs (stdout/stderr)", expanded=False):
    proc = st.session_state.ena3d_proc
    if proc is None:
        st.caption("No process started yet.")
    else:
        # Read whatever is currently available (non-blocking-ish for small output)
        try:
            if proc.stdout:
                st.session_state.ena3d_last_stdout += proc.stdout.read() or ""
            if proc.stderr:
                st.session_state.ena3d_last_stderr += proc.stderr.read() or ""
        except Exception:
            pass

        st.text_area("stdout", st.session_state.ena3d_last_stdout, height=150, key="ena3d_stdout")
        st.text_area("stderr", st.session_state.ena3d_last_stderr, height=150, key="ena3d_stderr")