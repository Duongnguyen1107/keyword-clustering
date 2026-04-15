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
    st.markdown("👉 **Chọn từ sidebar bên trái: _Keyword Clustering_**")

with col2:
    st.subheader("🔍 URL Classifier")
    st.write("Classify URL slugs từ GA4 export → niche + intent. Download enriched CSV cho dashboard.")
    st.markdown("👉 **Chọn từ sidebar bên trái: _Url Classifier_**")

st.divider()
st.info("💡 Dùng menu sidebar bên trái để chuyển giữa các tools")
st.caption("Model: all-MiniLM-L6-v2 · Built for Pinterest affiliate portfolio analysis")