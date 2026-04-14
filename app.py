import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime

st.set_page_config(page_title="Keyword Clustering", page_icon="🔑", layout="wide")

st.title("🔑 Keyword Clustering Tool")
st.caption("Powered by all-MiniLM-L6-v2 + AgglomerativeClustering")

# ── Sidebar controls ──
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Clustering Threshold", 0.70, 0.99, 0.82, 0.01,
                          help="Càng cao = cluster càng chặt, ít keywords/cluster hơn")
    sim_filter = st.slider("Similarity Filter", 0.70, 0.99, 0.88, 0.01,
                           help="Sub-keywords có similarity thấp hơn mức này sẽ tách ra singleton")
    batch_size = st.select_slider("Batch Size", options=[32, 64, 128], value=64)
    st.divider()
    st.markdown("**Hướng dẫn:**")
    st.markdown("1. Upload file Excel/CSV")
    st.markdown("2. Điều chỉnh settings nếu cần")
    st.markdown("3. Click **Run Clustering**")
    st.markdown("4. Download kết quả")

# ── File upload ──
uploaded = st.file_uploader("📂 Upload file keywords (.xlsx hoặc .csv)",
                             type=["xlsx", "xls", "csv"])

if uploaded:
    # Load preview
    try:
        if uploaded.name.endswith(".csv"):
            df_preview = pd.read_csv(uploaded, dtype=str, nrows=5)
        else:
            df_preview = pd.read_excel(uploaded, dtype=str, nrows=5)
        st.success(f"✅ Đã load: **{uploaded.name}**")
        st.dataframe(df_preview, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        st.stop()

    if st.button("🚀 Run Clustering", type="primary", use_container_width=True):
        # ── Load full data ──
        uploaded.seek(0)
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded, dtype=str)
        else:
            df = pd.read_excel(uploaded, dtype=str)

        df.columns = df.columns.str.strip()
        df = df.dropna(how='all')

        # Detect keyword column
        kw_col = None
        for c in df.columns:
            if c.lower() in ('keyword', 'keywords', 'kw', 'key', 'tu khoa', 'query'):
                kw_col = c
                break
        if kw_col is None:
            kw_col = df.columns[0]

        # Detect volume column
        vol_col = None
        for c in df.columns:
            if any(x in c.lower() for x in ('vol', 'volume', 'search', 'msv', 'sv')):
                vol_col = c
                break

        st.info(f"Keyword column: `{kw_col}` | Volume column: `{vol_col or 'không có'}`")

        # Extract keywords + volumes
        keywords = df[kw_col].dropna().str.strip().tolist()
        keywords = [k for k in keywords if k]
        seen = set(); unique_kws = []
        for k in keywords:
            if k not in seen:
                seen.add(k); unique_kws.append(k)
        keywords = unique_kws

        volumes = {}
        if vol_col:
            for _, row in df.iterrows():
                kw = str(row[kw_col]).strip() if pd.notna(row[kw_col]) else ''
                if not kw: continue
                try:
                    v = int(str(row[vol_col]).replace(',','').replace('.','').strip())
                except: v = 0
                if kw not in volumes: volumes[kw] = v

        st.write(f"**{len(keywords):,} unique keywords** sẽ được clustering")

        # ── Encode ──
        with st.spinner("⏳ Loading model + encoding keywords... (lần đầu ~30-60s)"):
            from sentence_transformers import SentenceTransformer
            @st.cache_resource
            def load_model():
                return SentenceTransformer('all-MiniLM-L6-v2')
            model = load_model()
            embeddings = model.encode(
                keywords, batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

        # ── Cluster ──
        with st.spinner("⏳ Clustering..."):
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics.pairwise import cosine_similarity

            distance_threshold = 1 - threshold
            N = len(keywords)

            if N > 3000:
                from sklearn.neighbors import kneighbors_graph
                connectivity = kneighbors_graph(
                    embeddings, n_neighbors=min(15, N-1),
                    metric='cosine', include_self=False, n_jobs=-1,
                )
                clustering = AgglomerativeClustering(
                    n_clusters=None, metric='cosine', linkage='complete',
                    distance_threshold=distance_threshold, connectivity=connectivity,
                )
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=None, metric='cosine', linkage='complete',
                    distance_threshold=distance_threshold,
                )
            labels = clustering.fit_predict(embeddings)

        # ── Build output ──
        with st.spinner("⏳ Building output..."):
            groups = {}
            for i, lbl in enumerate(labels):
                groups.setdefault(lbl, []).append(i)

            def pick_representative(members_with_vol):
                if len(members_with_vol) == 1: return members_with_vol[0][0]
                top3 = sorted(members_with_vol, key=lambda x: x[1], reverse=True)[:3]
                return min(top3, key=lambda x: len(x[0]))[0]

            final_data = []
            for lbl, idxs in sorted(groups.items(), key=lambda x: -len(x[1])):
                members = [(keywords[i], volumes.get(keywords[i], 0)) for i in idxs]
                cluster_vol = sum(v for _, v in members)
                main = pick_representative(members)
                main_global_idx = idxs[next(j for j,(k,_) in enumerate(members) if k==main)]
                main_emb = embeddings[main_global_idx].reshape(1,-1)

                subs = [(keywords[i], volumes.get(keywords[i],0), i)
                        for i in idxs if keywords[i] != main]
                sim_scores = {}
                if subs:
                    sub_embs = embeddings[[i for _,_,i in subs]]
                    sims = cosine_similarity(main_emb, sub_embs)[0]
                    for (kw,vol,_), sim in zip(subs, sims):
                        sim_scores[kw] = sim

                final_data.append({
                    'Chu de chinh': main, 'Tong Volume': cluster_vol,
                    'Cluster Size': len(idxs), 'Keyword': main,
                    'Volume': volumes.get(main,0), 'Is Main': 'YES',
                    'Similarity': '100%',
                })
                for kw, vol, _ in sorted(subs, key=lambda x: sim_scores.get(x[0],0), reverse=True):
                    sim = sim_scores.get(kw, 0)
                    if sim < sim_filter:
                        final_data.append({
                            'Chu de chinh': kw, 'Tong Volume': vol,
                            'Cluster Size': 1, 'Keyword': kw,
                            'Volume': vol, 'Is Main': 'YES', 'Similarity': '100%',
                        })
                    else:
                        final_data.append({
                            'Chu de chinh': main, 'Tong Volume': cluster_vol,
                            'Cluster Size': len(idxs), 'Keyword': kw,
                            'Volume': vol, 'Is Main': 'NO',
                            'Similarity': f"{sim*100:.1f}%",
                        })

            df_result = pd.DataFrame(final_data)
            df_result = df_result.sort_values(
                by=['Tong Volume', 'Volume'], ascending=[False, False]
            )

        # ── Stats ──
        n_clusters = len(set(labels))
        n_clustered = sum(1 for i,lbl in enumerate(labels)
                         if labels.tolist().count(lbl) > 1)
        coverage = n_clustered / len(keywords) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Keywords", f"{len(keywords):,}")
        col2.metric("Clusters", f"{n_clusters:,}")
        col3.metric("Clustered", f"{n_clustered:,}")
        col4.metric("Coverage", f"{coverage:.1f}%")

        # ── Preview ──
        st.subheader("📊 Preview kết quả (top 50 rows)")
        st.dataframe(df_result.head(50), use_container_width=True)

        # ── Download ──
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_result.to_excel(writer, index=False)
        buf.seek(0)

        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(
            label="⬇️ Download Excel",
            data=buf,
            file_name=f"Keyword_Clusters_{now}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )
else:
    st.info("👆 Upload file keywords để bắt đầu")
