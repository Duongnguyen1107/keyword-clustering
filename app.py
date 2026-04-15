import streamlit as st

st.set_page_config(
    page_title="Pinterest Affiliate Tools",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ Pinterest Affiliate Tools")
st.caption("Bộ công cụ phân tích keyword và URL cho affiliate sites")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔑 Keyword Clustering")
    st.write("Cluster keywords theo semantic similarity. Upload Excel/CSV → download kết quả phân nhóm.")
    st.page_link("pages/1_keyword_clustering.py", label="Mở tool →", icon="🔑")

with col2:
    st.subheader("🔍 URL Classifier")
    st.write("Classify URL slugs từ GA4 export → niche + intent bằng semantic embeddings. Download enriched CSV cho dashboard.")
    st.page_link("pages/2_url_classifier.py", label="Mở tool →", icon="🔍")

st.divider()
st.caption("Model: all-MiniLM-L6-v2 · Built for Pinterest affiliate portfolio analysis")