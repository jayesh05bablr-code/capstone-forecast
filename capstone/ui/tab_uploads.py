"""Upload company data tab: CSV/Excel (incl. Tracxn exports) or a Tracxn profile PDF."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from capstone import overnight, tracxn, tracxn_excel, uploads
from capstone.ui.common import XLSX_MIME, auto_update

PDF_FIELDS = {"market_cap_cr": "Market cap (₹ crore)", "pe": "P/E", "pb": "P/B",
              "eps_ttm": "EPS, trailing 12 months (₹)", "roe_pct": "ROE %",
              "debt_to_equity": "Debt to equity"}


def _tracxn_tables(data: bytes) -> list[tuple[str, pd.DataFrame]]:
    """Parse a Tracxn company export and let the user edit each table before saving."""
    smap = overnight.stock_map()
    parsed, notes = tracxn_excel.parse(data, smap["symbol"], dict(zip(smap["symbol"], smap["india_sector"])))
    st.success("Tracxn company export recognised. Check the tables below, edit anything "
               "that looks wrong, then save.")
    for note in notes:
        st.caption(note)
    ready = []
    for kind, df in parsed.items():
        st.markdown(f"**{kind.capitalize()}** ({len(df)} rows)")
        ready.append((kind, st.data_editor(df, num_rows="dynamic", key=f"tx_{kind}", use_container_width=True)))
    return ready


def _own_tables(tables: dict[str, pd.DataFrame]) -> list[tuple[str, pd.DataFrame]]:
    """Match each sheet of the user's own file to a template, and let them edit it."""
    kinds = list(uploads.TEMPLATES)
    ready = []
    for name, raw_df in tables.items():
        if name == "README" or raw_df.empty:
            continue
        std = uploads.standardise(raw_df)
        kind = uploads.detect(std)
        st.markdown(f"**{name}**: {len(raw_df)} rows, recognised columns: "
                    f"{', '.join(std.columns) or 'none'}")
        if kind is None:
            st.warning("Needs a symbol column plus at least one data column from the template.")
            continue
        kind = st.selectbox("Save as", kinds, index=kinds.index(kind), key=f"kind_{name}")
        if kind in ("fundamentals", "shareholding") and "as_of" not in std:
            st.warning("No date column found; add an as_of column (date the numbers were public).")
            continue
        ready.append((kind, st.data_editor(std, num_rows="dynamic", key=f"ed_{name}", use_container_width=True)))
    return ready


def _save(ready: list[tuple[str, pd.DataFrame]]) -> None:
    saved, failed = [], False
    for kind, df in ready:
        try:
            n = uploads.merge_save(kind, df)
            saved.append(f"Saved {n} {kind} rows.")
        except ValueError as exc:
            st.error(f"{kind}: {exc}")
            failed = True
    if saved and not failed:
        st.session_state["saved_msgs"] = saved
        auto_update("Saved " + ", ".join(k for k, _ in ready))
    for msg in saved:
        st.success(msg)


def _table_upload() -> None:
    st.subheader("1. Upload a CSV or Excel file (recommended)")
    st.markdown("The most accurate way to feed the model. A Tracxn company Excel export "
                "(TracxnExport-CompanyExcel-...xlsx) is read as it is. Your own file can hold fundamentals, "
                "shareholding (FII/DII), financials, investors and profile details, one sheet or table "
                "each. Column names are matched loosely (\"P/E\", \"PE Ratio\" and \"pe\" all work). "
                "Download the template to see the columns.")
    st.download_button("Download the Excel template", uploads.template_workbook(),
                       file_name="company_data_template.xlsx", mime=XLSX_MIME)
    with st.expander("What each sheet holds"):
        for kind, t in uploads.TEMPLATES.items():
            st.markdown(f"- **{kind}**: `{', '.join(t['columns'])}`. {t['about']}")
    up = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"], key="table_upload")
    if up is None:
        return
    try:
        tables = uploads.read_upload(up.name, up.getvalue())
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        tables = {}
    ready = _tracxn_tables(up.getvalue()) if tracxn_excel.is_tracxn(tables) else _own_tables(tables)
    if ready and st.button("Save uploaded data"):
        _save(ready)


def _pdf_upload(preds: pd.DataFrame, story: dict) -> None:
    st.subheader("2. Or upload a Tracxn profile PDF")
    st.markdown("Key numbers are pulled out for you to check and edit, then saved into "
                "`data/fundamentals.csv`. Saving retrains the model and updates the forecast automatically.")
    c1, c2 = st.columns([1, 2])
    symbol = c1.selectbox("Company", preds["symbol"], key="profile_symbol")
    as_of = c1.date_input("Data as of (report date)", value=date.today())
    pdf = c2.file_uploader("Tracxn profile (PDF)", type=["pdf"])
    if pdf is not None:
        data = pdf.getvalue()
        try:
            text = tracxn.pdf_text(data)
        except Exception as exc:
            st.error(f"Could not read this PDF ({type(exc).__name__}). Is it a scanned image?")
            text = ""
        if text:
            found = tracxn.extract(text, story.get("usdinr"))
            as_read = {k[4:]: v for k, v in found.items() if k.startswith("raw_")}
            if as_read:
                st.markdown("**Found in the PDF**")
                st.dataframe(pd.DataFrame(as_read.items(), columns=["Field", "Value"]), hide_index=True)
            else:
                st.info("No standard fields recognised; fill the numbers in below from the PDF.")
            st.markdown("**Numbers to save for the model** (edit anything that looks wrong)")
            edit = st.data_editor(
                pd.DataFrame({"Field": list(PDF_FIELDS.values()), "Value": [found.get(k) for k in PDF_FIELDS]}),
                disabled=["Field"], hide_index=True, key=f"edit_{pdf.name}")
            with st.expander("Text read from the PDF"):
                st.text(text[:5000])
            if st.button("Save to fundamentals"):
                values = dict(zip(PDF_FIELDS, pd.to_numeric(edit["Value"], errors="coerce")))
                sector = preds.set_index("symbol").loc[symbol, "india_sector"]
                tracxn.save_fundamentals(symbol, as_of, values, sector)
                tracxn.save_profile(symbol, pdf.name, data, found)
                prof_row = tracxn.profile_row(symbol, found)
                if prof_row:
                    uploads.merge_save("profile", pd.DataFrame([prof_row]))
                st.session_state["saved_msgs"] = [f"Saved {symbol} fundamentals as of {as_of}."]
                auto_update(f"Saved {symbol} from the PDF")
    if tracxn.PROFILES_FILE.exists():
        st.markdown("**Profiles uploaded so far**")
        st.dataframe(pd.read_csv(tracxn.PROFILES_FILE), use_container_width=True, hide_index=True)


def render(preds: pd.DataFrame, story: dict) -> None:
    _table_upload()
    _pdf_upload(preds, story)
