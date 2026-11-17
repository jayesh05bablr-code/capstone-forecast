"""Download data tab: every dataset behind the model in one Excel workbook."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from capstone import export
from capstone.ui.common import XLSX_MIME


def render() -> None:
    st.markdown("Everything behind the model in one Excel workbook, one sheet per dataset, with a "
                "README sheet explaining each one. Use it for your own EDA.")
    parts = export.sheets()
    st.dataframe(pd.DataFrame([(n, len(d), len(d.columns)) for n, d in parts.items()],
                              columns=["Sheet", "Rows", "Columns"]), hide_index=True)
    if st.button("Prepare Excel file"):
        with st.spinner("Building the workbook..."):
            st.session_state["xlsx"] = export.workbook()
    if "xlsx" in st.session_state:
        st.download_button("Download capstone_data.xlsx", st.session_state["xlsx"],
                           file_name=f"capstone_data_{date.today():%Y%m%d}.xlsx", mime=XLSX_MIME)
