# ena_tool.py
from __future__ import annotations
import re
import argparse
from pathlib import Path
from typing import List, Optional
from transformers import TrainerCallback
import pandas as pd


# For progress bar in streamlit while training
# Each time a training step ends, update the progress bar
class StreamlitProgressCallback(TrainerCallback):
    def __init__(self, total_steps, progress_bar):
        self.total_steps = max(int(total_steps or 0), 0)
        self.progress_bar = progress_bar

    def on_step_end(self, args, state, control, **kwargs):
        if not self.progress_bar or self.total_steps <= 0:
            return
        frac = min(state.global_step / self.total_steps, 1.0)
        self.progress_bar.progress(frac)


def detect_lang(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "en"

    zh_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    en_chars = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))

    if zh_chars == 0 and en_chars == 0:
        return "en"
    return "zh" if zh_chars >= en_chars else "en"



# Add some sanity checks for text column
def assert_has_usable_text(df: pd.DataFrame, text_col: str) -> None:
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in dataframe.")

    s = df[text_col].dropna().astype(str).str.strip()
    if s.empty or s.eq("").all() or s.str.lower().eq("nan").all():
        raise ValueError(f"Column '{text_col}' exists but contains no usable text.")


def split_zh_sentences(text: Optional[str], min_len: int = 8) -> List[str]:
    if text is None:
        return []
    text = str(text).strip()
    if not text or text.lower() == "nan":
        return []

    sents = re.split(r"[。！？；…]\s*|\n+", text)
    sents = [s.strip() for s in sents if s and s.strip()]

    if min_len and min_len > 0:
        sents = [s for s in sents if len(s) >= min_len]

    return sents


def split_en_sentences(text: Optional[str], min_len: int = 1) -> List[str]:
    if text is None:
        return []
    t = str(text).strip()
    if not t or t.lower() == "nan":
        return []

    t = re.sub(r"\s+", " ", t)

    abbr = r"(Mr|Ms|Mrs|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\."
    t = re.sub(abbr, lambda m: m.group(0).replace(".", "<DOT>"), t, flags=re.IGNORECASE)

    parts = re.split(r"[.!?;]\s*", t)
    sents = [p.replace("<DOT>", ".").strip() for p in parts if p and p.strip()]

    if min_len and min_len > 0:
        sents = [s for s in sents if len(s) >= min_len]

    return sents



def segment_csv(
    in_csv: Path,
    out_csv: Path,
    text_col: str,
    id_cols: List[str],
    group_col: Optional[str],
    lang: str,
    min_len_zh: int,
    min_len_en: int,
    keep_cols: Optional[List[str]] = None,
) -> None:
    df = pd.read_csv(in_csv)
    # assert_has_usable_text(df, text_col)

    # 1) handle duplicated columns
    if df.columns.duplicated().any():
        counts = {}
        new_cols = []
        for c in df.columns:
            if c not in counts:
                counts[c] = 0
                new_cols.append(c)
            else:
                counts[c] += 1
                new_cols.append(f"{c}__dup{counts[c]}")
        df.columns = new_cols

        # check required columns before using the text_col
    required_cols = [text_col, *id_cols]
    if group_col:
        required_cols.append(group_col)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing columns: {missing}. Existing columns: {list(df.columns)}"
        )
    
    assert_has_usable_text(df, text_col)

    # 2) use row_id as unique row identifier
    if "__row_id" in df.columns:

        df = df.rename(columns={"__row_id": "__row_id__orig"})
    df = df.reset_index(drop=False).rename(columns={"index": "__row_id"})
    row_id_col = "__row_id"


    # 4) segmentation
    def _segment_row(x):
        t = x.get(text_col)
        print("DEBUG raw text:", repr(t))

        if lang == "auto":
            l = detect_lang(str(t)) if pd.notna(t) else "en"
        else:
            l = lang

        if l == "zh":
            segs = split_zh_sentences(t, min_len=min_len_zh)
        else:
            segs = split_en_sentences(t, min_len=min_len_en)

        # if debug:
        #     print(f"DEBUG row_id={x[row_id_col]} lang={l} segments:", segs)

        return segs

    df["_segments"] = df.apply(_segment_row, axis=1)

    # print(f"Wrote segmented units: {out_csv} ({len(df_seg)} rows)")
    print("DEBUG text head:", df[text_col].head(3).tolist())
    print("DEBUG segments head:", df["_segments"].head(3).tolist())
    print("DEBUG non-empty segments:", sum(bool(s) for s in df["_segments"] if isinstance(s, list)))

    # 5) explode
    # 要保留的 meta 欄位：使用者選的 + 必需欄位
    base_must = [text_col, *id_cols]
    if group_col:
        base_must.append(group_col)

    # keep_cols=None → 保留全部欄位（最保險）
    if keep_cols is None:
        meta_cols = list(df.columns)
    else:
        # 只保留使用者選到且存在的欄位 + 必需欄位
        wish = list(dict.fromkeys(keep_cols + base_must))  # 去重保序
        meta_cols = [c for c in wish if c in df.columns]

    # 一定要帶上 row_id 與 _segments
    cols = list(dict.fromkeys(meta_cols + [row_id_col, "_segments"]))

    df_seg = (
        df[cols]
        .explode("_segments")
        .dropna(subset=["_segments"])
        .reset_index(drop=True)
        .rename(columns={"_segments": "text"})
    )

    # In case there is error
        # In case there is error: still write a CSV (at least headers) to avoid FileNotFound
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if df_seg.empty:
        df_seg = pd.DataFrame(columns=[c for c in cols if c!= "_segments"] + ["text"])
        df_seg.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"Wrote segmented units: {out_csv} ({len(df_seg)} rows)")
        return

    # 6) filter the length
    if lang == "auto":
        df_seg["lang"] = df_seg["text"].apply(detect_lang)
    else:
        df_seg["lang"] = lang

    # 7) segment_id, grouping by group_col if provided
    speaker_key = id_cols[0]
    if group_col:
        df_seg["segment_id"] = df_seg.groupby(group_col).cumcount() + 1
    else:
        df_seg["segment_id"] = range(1, len(df_seg) + 1)
       
    df_seg["unit_id"] = (
        df_seg[speaker_key].astype(str)
        + "_"
        + df_seg[row_id_col].astype(str)
        + "_"
        + df_seg["segment_id"].astype(str)
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_seg.to_csv(out_csv, index=False, encoding="utf-8-sig")



###### Paradigms and API #####################################################

# test_paradigms.py
from pathlib import Path
import pandas as pd
import yaml
from dataclasses import dataclass

@dataclass
class Concept:
    code: str
    label: str
    definition: str

def load_concepts_yaml(path: Path) -> list[Concept]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Concept(c["code"], c.get("label", c["code"]), c.get("definition", ""))
        for c in data.get("concepts", [])
    ]

def generate_paradigms_openai(concept: Concept, lang: str, n: int, model: str) -> list[str]:
    from openai import OpenAI
    client = OpenAI()

    if lang == "zh":
        lang_hint = "繁體中文"
    else:
        lang_hint = "English"

    prompt = f"""
Generate {n} representative example sentences for this concept.

Concept: {concept.label}
Definition: {concept.definition}

Language: {lang_hint}
One sentence per line.
""".strip()

    resp = client.responses.create(model=model, input=prompt)
    text = resp.output_text.strip()
    return [l.strip() for l in text.splitlines() if l.strip()]

def step_paradigms(
    concepts_yaml: Path,
    out_csv: Path,
    lang: str = "zh",
    n: int = 5,
    model: str = "gpt-5.2",
) -> None:
    concepts = load_concepts_yaml(concepts_yaml)
    rows = []

    for c in concepts:
        sents = generate_paradigms_openai(c, lang=lang, n=n, model=model)
        for s in sents:
            rows.append({
                "concept_code": c.code,
                "concept_label": c.label,
                "lang": lang,
                "text": s,
                "source": "gpt",
            })

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote paradigms to {out_csv}")

    ####################################Train Test####################################

from pathlib import Path
import pandas as pd
# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

def step_train_multiclass(
    paradigms_csv: Path,
    out_dir: Path,
    backbone: str = "bert",          
    epochs: int = 3,
    batch_size: int = 8,
    max_len: int = 128,
    seed: int = 42,
    progress_bar=None
) -> None:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from pathlib import Path
    import json

    model_name = {"bert": "bert-base-uncased", "roberta": "roberta-base"}[backbone]

    df = pd.read_csv(paradigms_csv).dropna(subset=["text", "concept_code"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    concepts = sorted(df["concept_code"].unique().tolist())
    code2id = {c: i for i, c in enumerate(concepts)}
    df["label"] = df["concept_code"].map(code2id).astype(int)

    

    id2code = {i: c for c, i in code2id.items()}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = {
        "code2id": code2id,
        "id2code": id2code,
    }

    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    train_df, val_df = train_test_split(
        df[["text", "label"]],
        test_size=0.2,
        random_state=seed,
        stratify=df["label"],
    )

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))


    total_steps = (len(train_ds) // batch_size) * epochs

    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    total_steps = (len(train_ds) // batch_size) * epochs


    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_len)

    train_ds = train_ds.map(tok, batched=True)
    val_ds = val_ds.map(tok, batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(concepts))

    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none",
        seed=seed,
    )

    callbacks = []
    if progress_bar is not None:
        from ena_tool import StreamlitProgressCallback  
        callbacks.append(StreamlitProgressCallback(total_steps, progress_bar))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
        
    

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics = {}
    metrics.update(train_result.metrics)
    metrics.update(eval_metrics)   

    return metrics
    

    print(f"Saved model to {out_dir}")
    return train_result.metrics

########Classify Test Predict#######################################

def step_predict_multiclass(
    units_csv: Path,
    model_dir: Path,
    out_csv: Path,
    text_col: str = "texts",
    batch_size: int = 16,
    max_len: int = 128,
) -> None:
    import json
    import torch
    import pandas as pd
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"[DEBUG] Reading units from: {units_csv.resolve()}")
    df = pd.read_csv(units_csv)
    if df.empty:
        print("[STOP] Prediction skipped: no units to predict.")
        return

    model_dir = Path(model_dir)

    label_map = json.loads((model_dir / "label_map.json").read_text(encoding="utf-8"))
    code2id = label_map["code2id"]
    id2code = {int(k): v for k, v in label_map["id2code"].items()} if isinstance(next(iter(label_map["id2code"])), str) else label_map["id2code"]
    

    
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    df = pd.read_csv(units_csv)

    print(f"[DEBUG] Units rows: {len(df)}")
    print(f"[DEBUG] Units columns: {list(df.columns)}")
    texts = df[text_col].astype(str).tolist()

    print(f"[DEBUG] Number of texts: {len(texts)}")

    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    probs = np.vstack(all_probs)

    if not all_probs:
        print("[STOP] No predictions were generated because there were no input texts.")
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        return
    

    code2label = label_map.get("code2label", {})  # e.g., {"C1": "claim", ...}

    def clean(s: str) -> str:
        return str(s).strip().replace(" ", "_")

    for idx, code in id2code.items():
        label = code2label.get(code, code)  
        df[f"p_{clean(label)}"] = probs[:, idx]


    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Wrote predictions to: {out_csv} ({len(df)} rows)")
    return

#########binarize ENA#######################################
import numpy as np

def step_binarize_ena(
    in_csv: Path,
    out_csv: Path,
    mode: str,
    threshold: float,
    prob_prefix: str,
    keep_cols: list[str],
) -> None:
    df = pd.read_csv(in_csv)

    missing_keep = [c for c in keep_cols if c not in df.columns]
    if missing_keep:
        raise ValueError(f"Missing keep_cols in input: {missing_keep}. existing: {list(df.columns)}")

    prob_cols = [c for c in df.columns if c.startswith(prob_prefix)]
    if not prob_cols:
        raise ValueError(f"No probability columns found with prefix '{prob_prefix}' in {in_csv}")

    prob_cols = sorted(prob_cols)  
    probs = df[prob_cols].to_numpy()

    if mode == "top1":
        top_idx = probs.argmax(axis=1)
        coded = np.zeros_like(probs, dtype=int)
        coded[np.arange(len(df)), top_idx] = 1
    elif mode == "threshold":
        coded = (probs >= threshold).astype(int)
    else:
        raise ValueError("mode must be one of: top1, threshold")

    import json
    label_map_path = Path(in_csv).parent / "label_map.json"
    code2label = {}
    if label_map_path.exists():
        lm = json.loads(label_map_path.read_text(encoding="utf-8"))
        code2label = lm.get("code2label", {})

    concept_cols = [
        code2label.get(c.replace(prob_prefix, "", 1), c.replace(prob_prefix, "", 1))
        for c in prob_cols
    ]
    coded_df = pd.DataFrame(coded, columns=concept_cols)

    out_df = pd.concat([df[keep_cols].reset_index(drop=True), coded_df.reset_index(drop=True)], axis=1)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote ENA-coded CSV to: {out_csv}")




#######################################

def main():
    p = argparse.ArgumentParser(description="ENA local tool — CSV segmentation + GPT paradigms")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---------- segment_csv ----------
    s = sub.add_parser("segment_csv", help="Segment a CSV text column into units (explode)")
    s.add_argument("--in_csv", required=True, type=Path)
    s.add_argument("--out_csv", required=True, type=Path)
    s.add_argument("--text_col", required=True, help="column name: the text to be segmented")
    s.add_argument("--id_cols", required=True, help="reserved id columns, e.g. _id,student_id")
    s.add_argument("--group_col", default=None, help="segment numbering group by this column")
    s.add_argument("--lang", default="auto", choices=["auto", "zh", "en"])
    s.add_argument("--min_len_zh", type=int, default=8)
    s.add_argument("--min_len_en", type=int, default=1)

    # ---------- paradigms ----------
    g = sub.add_parser("paradigms", help="Generate concept paradigm sentences via OpenAI API")
    g.add_argument("--concepts", required=True, type=Path, help="path to concepts.yaml")
    g.add_argument("--out_csv", required=True, type=Path, help="output paradigms csv")
    g.add_argument("--lang", required=True, choices=["zh", "en"], help="language for paradigms")
    g.add_argument("--n", type=int, default=20, help="number of paradigms per concept")
    g.add_argument("--model", type=str, default="gpt-5.2", help="OpenAI model name")
        # ---------- train_multiclass ----------
    t = sub.add_parser("train_multiclass", help="Train a multi-class classifier from paradigms.csv")
    t.add_argument("--paradigms_csv", required=True, type=Path, help="input paradigms csv")
    t.add_argument("--out_dir", required=True, type=Path, help="output model directory")
    t.add_argument("--backbone", required=True, choices=["bert", "roberta"], help="model backbone")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--max_len", type=int, default=128)
    t.add_argument("--seed", type=int, default=42)
        # ---------- predict_multiclass ----------
    p = sub.add_parser("predict_multiclass", help="Predict concept probabilities for segmented units")
    p.add_argument("--units_csv", required=True, type=Path)
    p.add_argument("--model_dir", required=True, type=Path)
    p.add_argument("--out_csv", required=True, type=Path)
    p.add_argument("--text_col", default="text")
        # ---------- binarize_ena ----------
    b = sub.add_parser("binarize_ena", help="Convert p_* probabilities into ENA 0/1 concept columns")
    b.add_argument("--in_csv", required=True, type=Path)
    b.add_argument("--out_csv", required=True, type=Path)
    b.add_argument("--mode", required=True, choices=["top1", "threshold"])
    b.add_argument("--threshold", type=float, default=0.5)
    b.add_argument("--prob_prefix", default="p_", help="prefix for probability columns (default: p_)")
    b.add_argument(
        "--keep_cols",
        default="unit_id,student_id,segment_id,text",
        help="comma-separated metadata columns to keep",
    )
    args = p.parse_args()

    if args.cmd == "segment_csv":
        segment_csv(
            in_csv=args.in_csv,
            out_csv=args.out_csv,
            text_col=args.text_col,
            id_cols=[c.strip() for c in args.id_cols.split(",") if c.strip()],
            group_col=args.group_col,
            lang=args.lang,
            min_len_zh=args.min_len_zh,
            min_len_en=args.min_len_en,
        )

    elif args.cmd == "paradigms":
        step_paradigms(
            concepts_yaml=args.concepts,
            out_csv=args.out_csv,
            lang=args.lang,
            n=args.n,
            model=args.model,
        )
    elif args.cmd == "train_multiclass":
        step_train_multiclass(
            paradigms_csv=args.paradigms_csv,
            out_dir=args.out_dir,
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_len=args.max_len,
            seed=args.seed,
        )

    elif args.cmd == "predict_multiclass":
        step_predict_multiclass(
            units_csv=args.units_csv,
            model_dir=args.model_dir,
            out_csv=args.out_csv,
            text_col=args.text_col,
        )

    elif args.cmd == "binarize_ena":
        step_binarize_ena(
            in_csv=args.in_csv,
            out_csv=args.out_csv,
            mode=args.mode,
            threshold=args.threshold,
            prob_prefix=args.prob_prefix,
            keep_cols=[c.strip() for c in args.keep_cols.split(",") if c.strip()],
        )

        

if __name__ == "__main__":
    main()


