"""
Auto Data Analyzer v2
Upload a CSV/Excel file -> clean, validate, visualize, get insights, export.

Run:  streamlit run app.py
"""
import io
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Auto Data Analyzer", page_icon="📊", layout="wide")


# =====================================================================
# Helpers
# =====================================================================
def show(fig):
    """Display a plotly chart (works on old and new Streamlit versions)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def clean_column_name(name, used):
    new = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")
    if not new:
        new = "column"
    base, i = new, 2
    while new in used:
        new = f"{base}_{i}"
        i += 1
    used.add(new)
    return new


def dataset_stats(df):
    return {
        "Rows": len(df),
        "Columns": df.shape[1],
        "Missing cells": int(df.isna().sum().sum()),
        "Duplicate rows": int(df.duplicated().sum()),
    }


def to_numeric_text(s):
    """Convert text like 'PKR 1,200' or '45%' to numbers (NaN if impossible)."""
    t = s.astype(str).str.strip()
    t = t.str.replace(r"(?i)^(pkr|rs\.?|usd|inr)\s*", "", regex=True)
    t = t.str.replace(r"[,\s%$€£₹]", "", regex=True)
    return pd.to_numeric(t, errors="coerce")


def try_datetime(s, dayfirst):
    sample = s.dropna().astype(str)
    if sample.empty or not sample.str.contains(r"[-/:]").mean() > 0.8:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
    return parsed


# =====================================================================
# Cleaning
# =====================================================================
def clean_data(df, o):
    """o = dict of options from the sidebar. Returns (cleaned_df, log_list)."""
    log = []
    d = df.copy()

    # 1) column names
    if o["clean_names"]:
        used = set()
        old = list(d.columns)
        d.columns = [clean_column_name(c, used) for c in old]
        changed = sum(a != b for a, b in zip(old, d.columns))
        log.append(f"Standardized {changed} column name(s) (lowercase, no special characters).")

    # 2) whitespace + empty strings -> NaN
    for c in d.select_dtypes(include=["object", "string"]).columns:
        d[c] = d[c].apply(lambda v: v.strip() if isinstance(v, str) else v)
        d[c] = d[c].replace({"": np.nan, "nan": np.nan, "NaN": np.nan,
                             "NULL": np.nan, "null": np.nan, "N/A": np.nan, "n/a": np.nan})
    log.append("Trimmed whitespace and converted blank/NULL/N/A text to missing values.")

    # 3) smart type detection
    if o["detect_types"]:
        for c in d.select_dtypes(include=["object", "string"]).columns:
            nonnull = d[c].notna().sum()
            if nonnull == 0:
                continue
            num = to_numeric_text(d[c].where(d[c].notna()))
            if num.notna().sum() / nonnull >= 0.8:
                d[c] = num
                log.append(f"Column '{c}': converted text to numbers.")
                continue
            dt = try_datetime(d[c], o["dayfirst"])
            if dt is not None and dt.notna().sum() / nonnull >= 0.8:
                d[c] = dt
                log.append(f"Column '{c}': converted text to dates.")

    # 4) duplicates
    if o["drop_dupes"]:
        n = int(d.duplicated().sum())
        d = d.drop_duplicates()
        log.append(f"Removed {n} duplicate row(s).")

    # 5) drop columns with too many missing
    if o["col_missing_limit"] < 100:
        pct = d.isna().mean() * 100
        drop = pct[pct > o["col_missing_limit"]].index.tolist()
        if drop:
            d = d.drop(columns=drop)
            log.append(f"Dropped column(s) with >{o['col_missing_limit']}% missing: {', '.join(drop)}")

    # 6) missing values
    strat = o["missing"]
    if strat == "Drop rows with missing":
        n = len(d)
        d = d.dropna()
        log.append(f"Dropped {n - len(d)} row(s) containing missing values.")
    elif strat != "Leave as is":
        for c in d.columns:
            k = int(d[c].isna().sum())
            if k == 0:
                continue
            if pd.api.types.is_numeric_dtype(d[c]):
                if strat == "Mean":
                    val, name = d[c].mean(), "mean"
                else:
                    val, name = d[c].median(), "median"
            else:
                m = d[c].mode()
                val, name = (m.iloc[0] if not m.empty else "Unknown"), "most frequent value"
            d[c] = d[c].fillna(val)
            log.append(f"Column '{c}': filled {k} missing value(s) with {name} ({val}).")

    # 7) outliers
    method = o["outlier"]
    if method != "None":
        num_cols = d.select_dtypes(include=np.number).columns
        if method.startswith("IQR"):
            lo_hi = {}
            for c in num_cols:
                q1, q3 = d[c].quantile([0.25, 0.75])
                iqr = q3 - q1
                lo_hi[c] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        else:
            lo_hi = {}
            for c in num_cols:
                mu, sd = d[c].mean(), d[c].std()
                if pd.notna(sd) and sd > 0:
                    lo_hi[c] = (mu - o["z"] * sd, mu + o["z"] * sd)
        if "Cap" in method:
            for c, (lo, hi) in lo_hi.items():
                n = int(((d[c] < lo) | (d[c] > hi)).sum())
                if n:
                    d[c] = d[c].clip(lo, hi)
                    log.append(f"Column '{c}': capped {n} outlier(s) to [{lo:.2f}, {hi:.2f}].")
        else:
            mask = pd.Series(False, index=d.index)
            for c, (lo, hi) in lo_hi.items():
                mask |= (d[c] < lo) | (d[c] > hi)
            log.append(f"Removed {int(mask.sum())} row(s) with outliers ({method}).")
            d = d[~mask]

    d = d.reset_index(drop=True)
    return d, log


# =====================================================================
# Validation rules
# =====================================================================
def violations(df, rule):
    """Return boolean Series: True where a row violates the rule."""
    col, kind = rule["column"], rule["type"]
    if col not in df.columns:
        return None
    s = df[col]
    if kind == "Not null":
        return s.isna()
    if kind == "Unique":
        return s.duplicated(keep=False) & s.notna()
    if kind in ("Minimum value", "Maximum value", "Between"):
        x = pd.to_numeric(s, errors="coerce")
        if kind == "Minimum value":
            return x < rule["v1"]
        if kind == "Maximum value":
            return x > rule["v1"]
        return (x < rule["v1"]) | (x > rule["v2"])
    if kind == "Allowed values":
        allowed = {a.strip() for a in rule["text"].split(",") if a.strip()}
        return s.notna() & ~s.astype(str).isin(allowed)
    if kind == "Matches regex":
        return s.notna() & ~s.astype(str).str.match(rule["text"])
    return None


def rule_label(r):
    k = r["type"]
    if k == "Minimum value":
        return f"{r['column']} >= {r['v1']}"
    if k == "Maximum value":
        return f"{r['column']} <= {r['v1']}"
    if k == "Between":
        return f"{r['column']} between {r['v1']} and {r['v2']}"
    if k in ("Allowed values", "Matches regex"):
        return f"{r['column']} {k.lower()}: {r['text']}"
    return f"{r['column']}: {k.lower()}"


# =====================================================================
# Auto insights
# =====================================================================
def generate_insights(d):
    out = []
    out.append(f"The dataset has **{len(d):,} rows** and **{d.shape[1]} columns**.")

    miss = (d.isna().mean() * 100).round(1)
    for c, p in miss[miss > 0].sort_values(ascending=False).items():
        out.append(f"Column **{c}** has **{p}%** missing values.")

    num = d.select_dtypes(include=np.number)
    if num.shape[1] > 1:
        corr = num.corr().abs()
        seen = set()
        for a in corr.columns:
            for b in corr.columns:
                if a != b and (b, a) not in seen and corr.loc[a, b] >= 0.7:
                    seen.add((a, b))
                    r = num[a].corr(num[b])
                    kind = "positively" if r > 0 else "negatively"
                    out.append(f"**{a}** and **{b}** are strongly {kind} correlated (r = {r:.2f}).")

    for c in num.columns:
        s = num[c].dropna()
        if len(s) >= 8:
            sk = s.skew()
            if abs(sk) > 1:
                out.append(f"**{c}** is highly skewed ({'right' if sk > 0 else 'left'}, skew = {sk:.2f}); "
                           "consider a log transform.")
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            n = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            if n:
                out.append(f"**{c}** has **{n}** potential outlier(s) (IQR rule).")
        if s.nunique() <= 1 and len(s):
            out.append(f"**{c}** is constant and adds no information.")

    for c in d.select_dtypes(exclude=[np.number, "datetime"]).columns:
        vc = d[c].value_counts(normalize=True)
        if len(vc) and len(d) >= 8 and vc.iloc[0] > 0.6 and len(vc) > 1:
            out.append(f"In **{c}**, the value '{vc.index[0]}' makes up {vc.iloc[0] * 100:.0f}% of rows.")
        if d[c].nunique() == len(d) and len(d) > 5:
            out.append(f"**{c}** has a unique value in every row (likely an ID column).")

    for c in d.select_dtypes(include="datetime").columns:
        out.append(f"**{c}** spans **{d[c].min().date()}** to **{d[c].max().date()}**.")

    if len(num.columns):
        top = num.sum().idxmax()
        out.append(f"**{top}** has the largest total ({num[top].sum():,.2f}).")
    return out


# =====================================================================
# Export helpers
# =====================================================================
def to_excel_bytes(df, log):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Cleaned data")
        pd.DataFrame({"Cleaning log": log}).to_excel(w, index=False, sheet_name="Cleaning log")
    return buf.getvalue()


def log_text(log, before, after, file_name):
    lines = [f"Cleaning log - {file_name}", f"Generated: {datetime.now():%Y-%m-%d %H:%M}", "",
             "BEFORE: " + ", ".join(f"{k}={v}" for k, v in before.items()),
             "AFTER : " + ", ".join(f"{k}={v}" for k, v in after.items()), "", "Steps:"]
    lines += [f"{i}. {t}" for i, t in enumerate(log, 1)]
    return "\n".join(lines)


# =====================================================================
# Loading
# =====================================================================
@st.cache_data(show_spinner=False)
def load_file(data: bytes, name: str, sheet):
    if name.lower().endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(data))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(data), encoding="latin-1")
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet)


# =====================================================================
# UI
# =====================================================================
def sidebar_options():
    st.sidebar.header("⚙️ Cleaning settings")
    o = {}
    o["clean_names"] = st.sidebar.checkbox("Standardize column names", True)
    o["detect_types"] = st.sidebar.checkbox("Auto-detect numbers & dates stored as text", True)
    o["dayfirst"] = st.sidebar.checkbox("Dates are day-first (e.g. 25-12-2024)", True,
                                        disabled=not o["detect_types"])
    o["drop_dupes"] = st.sidebar.checkbox("Drop duplicate rows", True)
    o["col_missing_limit"] = st.sidebar.slider("Drop columns with missing % above", 0, 100, 100,
                                               help="100 = never drop columns")
    o["missing"] = st.sidebar.selectbox(
        "Missing values", ["Median", "Mean", "Drop rows with missing", "Leave as is"],
        help="Numeric columns use median/mean; text columns use the most frequent value.")
    o["outlier"] = st.sidebar.selectbox(
        "Outlier handling",
        ["None", "IQR - Cap", "IQR - Remove rows", "Z-score - Cap", "Z-score - Remove rows"])
    o["z"] = 3.0
    if o["outlier"].startswith("Z-score"):
        o["z"] = st.sidebar.slider("Z-score threshold", 1.5, 4.0, 3.0, 0.1)
    return o


def tab_overview(raw, cleaned, log, before, after):
    st.subheader("Before vs after cleaning")
    cols = st.columns(4)
    for col, key in zip(cols, before):
        col.metric(key, after[key], after[key] - before[key], delta_color="off")
    st.caption("The small number under each value is the change caused by cleaning.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Raw data**")
        st.dataframe(raw.head(100))
    with c2:
        st.markdown("**Cleaned data**")
        st.dataframe(cleaned.head(100))

    st.subheader("What was changed")
    for t in log:
        st.write(f"- {t}")

    st.subheader("Column summary")
    summary = pd.DataFrame({
        "dtype": cleaned.dtypes.astype(str),
        "missing": cleaned.isna().sum(),
        "missing %": (cleaned.isna().mean() * 100).round(2),
        "unique": cleaned.nunique(),
    })
    st.dataframe(summary)
    st.markdown("**Statistics**")
    st.dataframe(cleaned.describe(include="all").T.fillna("").astype(str))


def tab_validation(raw, cleaned):
    st.subheader("Automatic checks (on raw data)")
    issues = []
    if raw.duplicated().sum():
        issues.append(f"{int(raw.duplicated().sum())} duplicate rows")
    for c in raw.columns[raw.isna().mean() > 0.3]:
        issues.append(f"'{c}' has more than 30% missing values")
    for c in raw.columns[raw.nunique() <= 1]:
        issues.append(f"'{c}' is constant")
    for c in raw.select_dtypes(include=np.number):
        if (raw[c] < 0).any():
            issues.append(f"'{c}' contains negative values (check if valid)")
    if issues:
        for i in issues:
            st.warning(i)
    else:
        st.success("No major issues found.")

    st.subheader("Custom validation rules (run on cleaned data)")
    if "rules" not in st.session_state:
        st.session_state.rules = []

    with st.form("rule_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        column = c1.selectbox("Column", list(cleaned.columns))
        kind = c2.selectbox("Rule", ["Minimum value", "Maximum value", "Between", "Not null",
                                     "Unique", "Allowed values", "Matches regex"])
        c3, c4, c5 = st.columns(3)
        v1 = c3.number_input("Value / lower bound", value=0.0)
        v2 = c4.number_input("Upper bound (Between only)", value=100.0)
        text = c5.text_input("Allowed values (a,b,c) or regex")
        if st.form_submit_button("➕ Add rule"):
            st.session_state.rules.append(
                {"column": column, "type": kind, "v1": v1, "v2": v2, "text": text})

    if st.session_state.rules:
        results = []
        for i, r in enumerate(st.session_state.rules):
            bad = violations(cleaned, r)
            if bad is None:
                results.append({"Rule": rule_label(r), "Status": "⚠️ column not found", "Violations": "-"})
                continue
            n = int(bad.sum())
            results.append({"Rule": rule_label(r), "Status": "✅ Pass" if n == 0 else "❌ Fail",
                            "Violations": n})
            if n:
                with st.expander(f"Rows violating: {rule_label(r)} ({n})"):
                    st.dataframe(cleaned[bad].head(200))
        st.dataframe(pd.DataFrame(results))
        if st.button("Clear all rules"):
            st.session_state.rules = []
            st.rerun()
    else:
        st.info("No rules yet. Example: column 'age', rule 'Between', 0 and 100.")


def tab_visualization(d):
    num = d.select_dtypes(include=np.number).columns.tolist()
    cat = d.select_dtypes(exclude=[np.number, "datetime"]).columns.tolist()
    dates = d.select_dtypes(include="datetime").columns.tolist()

    kinds = ["Histogram", "Box plot", "Bar chart", "Pie chart", "Scatter", "Correlation heatmap",
             "Line chart (dates)", "Pair plot"]
    kind = st.selectbox("Chart type", kinds)

    if kind == "Histogram":
        if not num:
            return st.info("No numeric columns.")
        c = st.selectbox("Column", num)
        color = st.selectbox("Color by (optional)", ["None"] + cat)
        show(px.histogram(d, x=c, color=None if color == "None" else color))
    elif kind == "Box plot":
        if not num:
            return st.info("No numeric columns.")
        y = st.selectbox("Numeric column", num)
        x = st.selectbox("Group by (optional)", ["None"] + cat)
        show(px.box(d, y=y, x=None if x == "None" else x, points="outliers"))
    elif kind == "Bar chart":
        if not cat:
            return st.info("No text/category columns.")
        c = st.selectbox("Category column", cat)
        agg_col = st.selectbox("Value (default = count)", ["Count"] + num)
        n = st.slider("Top N", 3, 50, 15)
        if agg_col == "Count":
            t = d[c].value_counts().head(n).reset_index()
            t.columns = [c, "count"]
            show(px.bar(t, x=c, y="count"))
        else:
            t = d.groupby(c)[agg_col].sum().sort_values(ascending=False).head(n).reset_index()
            show(px.bar(t, x=c, y=agg_col))
    elif kind == "Pie chart":
        if not cat:
            return st.info("No text/category columns.")
        c = st.selectbox("Category column", cat)
        n = st.slider("Top N slices", 3, 15, 8)
        t = d[c].value_counts().head(n).reset_index()
        t.columns = [c, "count"]
        show(px.pie(t, names=c, values="count"))
    elif kind == "Scatter":
        if len(num) < 2:
            return st.info("Need at least 2 numeric columns.")
        x = st.selectbox("X", num, index=0)
        y = st.selectbox("Y", num, index=1)
        color = st.selectbox("Color by (optional)", ["None"] + cat)
        show(px.scatter(d, x=x, y=y, color=None if color == "None" else color))
    elif kind == "Correlation heatmap":
        if len(num) < 2:
            return st.info("Need at least 2 numeric columns.")
        show(px.imshow(d[num].corr(), text_auto=".2f", zmin=-1, zmax=1,
                       color_continuous_scale="RdBu_r"))
    elif kind == "Line chart (dates)":
        if not dates or not num:
            return st.info("Needs a date column and a numeric column. "
                           "Turn on 'Auto-detect numbers & dates' in the sidebar.")
        x = st.selectbox("Date column", dates)
        y = st.selectbox("Value", num)
        agg = st.selectbox("If dates repeat, combine by", ["None", "mean", "sum"])
        t = d[[x, y]].dropna().sort_values(x)
        if agg != "None":
            t = t.groupby(x, as_index=False)[y].agg(agg)
        show(px.line(t, x=x, y=y, markers=True))
    elif kind == "Pair plot":
        if len(num) < 2:
            return st.info("Need at least 2 numeric columns.")
        picked = st.multiselect("Columns (max 6)", num, default=num[:4], max_selections=6)
        if len(picked) >= 2:
            show(px.scatter_matrix(d, dimensions=picked))


def tab_insights(d):
    st.subheader("Auto insights")
    for t in generate_insights(d):
        st.markdown(f"- {t}")


def tab_export(cleaned, log, before, after, file_name, sig):
    st.subheader("Download")
    stem = file_name.rsplit(".", 1)[0]
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ Cleaned CSV", cleaned.to_csv(index=False).encode("utf-8"),
                       f"{stem}_cleaned.csv", "text/csv")
    c2.download_button("⬇️ Cleaned Excel (+ log sheet)", to_excel_bytes(cleaned, log),
                       f"{stem}_cleaned.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c3.download_button("⬇️ Cleaning log (.txt)", log_text(log, before, after, file_name),
                       f"{stem}_cleaning_log.txt", "text/plain")

    st.subheader("Full profiling report")
    st.caption("Generated with ydata-profiling (minimal mode). Can take a while on large files.")
    if st.button("Generate profiling report"):
        try:
            from ydata_profiling import ProfileReport
            with st.spinner("Building report..."):
                html = ProfileReport(cleaned, minimal=True).to_html()
            st.session_state["profile"] = (sig, html)
        except ImportError:
            st.error("ydata-profiling is not installed. Run: pip install ydata-profiling")
        except Exception as e:
            st.error(f"Could not build report: {e}")

    saved = st.session_state.get("profile")
    if saved and saved[0] == sig:
        st.download_button("⬇️ Profiling report (.html)", saved[1].encode("utf-8"),
                           f"{stem}_profile.html", "text/html")
        components.html(saved[1], height=800, scrolling=True)


def run_app():
    st.title("📊 Auto Data Analyzer")
    st.caption("Upload once. Get cleaning, validation, charts, insights and exports.")
    opts = sidebar_options()

    file = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "xls"])
    if not file:
        st.info("Upload a CSV or Excel file to begin.")
        return

    data = file.getvalue()
    sheet = 0
    if not file.name.lower().endswith(".csv"):
        try:
            sheets = pd.ExcelFile(io.BytesIO(data)).sheet_names
        except Exception as e:
            return st.error(f"Could not read Excel file: {e}")
        if len(sheets) > 1:
            sheet = st.selectbox("Sheet", sheets)
        else:
            sheet = sheets[0]

    try:
        raw = load_file(data, file.name, sheet)
    except Exception as e:
        return st.error(f"Could not read file: {e}")

    if raw.empty:
        return st.warning("The file has no data.")

    cleaned, log = clean_data(raw, opts)
    if cleaned.empty:
        st.warning("Cleaning removed every row. Loosen the settings in the sidebar.")
        return

    before, after = dataset_stats(raw), dataset_stats(cleaned)
    sig = f"{file.name}|{sheet}|{sorted(opts.items())}"
    st.success(f"Loaded {raw.shape[0]} rows and {raw.shape[1]} columns "
               f"→ {cleaned.shape[0]} rows and {cleaned.shape[1]} columns after cleaning.")

    t1, t2, t3, t4, t5 = st.tabs(["Overview & Cleaning", "Validation", "Visualization",
                                  "Insights", "Export"])
    with t1:
        tab_overview(raw, cleaned, log, before, after)
    with t2:
        tab_validation(raw, cleaned)
    with t3:
        tab_visualization(cleaned)
    with t4:
        tab_insights(cleaned)
    with t5:
        tab_export(cleaned, log, before, after, file.name, sig)


def footer():
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:gray; padding:10px 0;'>"
        "Made by <b>Kumar Harsh</b> | "
        "<a href='mailto:kh949118@gmail.com'>kh949118@gmail.com</a>"
        "</div>",
        unsafe_allow_html=True,
    )


def main():
    run_app()
    footer()


if __name__ == "__main__":
    main()