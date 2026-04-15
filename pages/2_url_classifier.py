import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


st.title("🔍 URL Semantic Classifier")
st.caption("Upload GA4 CSV → classify niche + intent bằng semantic embeddings → download enriched CSV")

# ── Prototypes ───────────────────────────────────────────────────────────────
NICHE_PROTOTYPES = {
    "Home Decor": [
        "bedroom decor ideas for small spaces",
        "living room furniture and decoration aesthetic",
        "bathroom vanity mirror wall art ideas",
        "entryway hallway rug shelf organization",
        "boho farmhouse modern minimalist interior design",
        "curtains drapes window treatment home styling",
        "throw pillow blanket cozy home aesthetic",
        "home organization storage basket bin declutter",
        "wall decor frame gallery art print",
        "couch sofa accent chair living room furniture",
        "chandelier pendant lamp sconce lighting",
        "dresser nightstand bed frame bedroom furniture",
    ],
    "Garden/Outdoor": [
        "backyard garden landscaping ideas design",
        "patio outdoor furniture seating decoration",
        "raised garden bed vegetable herb planting",
        "flower plant succulent indoor outdoor pot",
        "lawn care grass maintenance tips",
        "outdoor string lights patio ambiance",
        "garden tools planting watering hose",
        "pergola deck fence outdoor structure",
    ],
    "Kitchen": [
        "kitchen organization storage solutions countertop",
        "cookware pan pot set kitchen tools",
        "coffee maker espresso machine kitchen appliance",
        "knife set cutting board kitchen gadget",
        "kitchen shelf cabinet pantry organizer",
        "air fryer instant pot slow cooker pressure cooker",
        "kitchen decor aesthetic farmhouse modern",
        "toaster blender food processor small appliance",
    ],
    "Food/Recipe": [
        "easy chicken dinner recipe weeknight family",
        "pasta soup salad healthy meal prep ideas",
        "beef pork salmon seafood cooking dinner",
        "keto vegan vegetarian gluten free healthy eating",
        "crockpot slow cooker casserole one pot recipe",
        "sauce marinade dressing seasoning homemade",
        "30 minute quick easy dinner ideas",
        "comfort food hearty filling dinner recipe",
    ],
    "Food/Baking": [
        "chocolate cake cupcake dessert recipe from scratch",
        "cookie brownie bar baking easy beginner",
        "sourdough bread muffin scone pastry baking",
        "frosting buttercream icing fondant cake decorating",
        "cheesecake no bake dessert easy recipe",
        "birthday cake ideas decoration tutorial",
        "pie tart galette pastry shell filling",
    ],
    "Styling": [
        "outfit ideas what to wear casual everyday",
        "fashion style clothing aesthetic look inspiration",
        "dress jeans boots sneakers shoes styling",
        "capsule wardrobe minimalist fashion basics",
        "summer winter fall spring seasonal outfit",
        "bag purse handbag accessory jewelry styling",
        "how to style outfit ideas for women",
    ],
    "Hair/Beauty": [
        "hairstyle haircut hair color ideas inspiration",
        "blonde brunette highlights balayage color ideas",
        "nail art design manicure gel ideas",
        "skincare routine steps products morning night",
        "makeup tutorial look beginner natural glam",
        "curtain bangs layers pixie bob haircut",
        "lashes brows glam beauty look",
    ],
    "Tattoo": [
        "tattoo design ideas inspiration placement",
        "small fine line minimalist tattoo art",
        "sleeve floral geometric mandala tattoo",
        "meaningful symbol quote tattoo ideas",
        "tattoo aftercare healing moisturizer",
        "watercolor blackwork traditional tattoo style",
    ],
    "Wedding/Event": [
        "wedding decoration ceremony reception ideas",
        "bridal shower bachelorette party ideas themes",
        "engagement proposal anniversary romantic ideas",
        "baby shower gender reveal party decoration",
        "birthday party table decoration theme setup",
        "wedding floral arrangement centerpiece bouquet",
        "DIY wedding craft decoration handmade budget",
    ],
    "DIY/Craft": [
        "DIY home project tutorial step by step beginner",
        "crochet knitting sewing pattern handmade craft",
        "woodworking build shelf furniture project",
        "painting canvas art craft kids activity",
        "repurpose upcycle thrift flip makeover project",
        "macrame wreath candle making craft project",
    ],
    "Lifestyle": [
        "morning routine productivity self care habits",
        "travel destination guide bucket list tips",
        "minimalist lifestyle wellness mental health",
        "personal finance budget saving money tips",
    ],
    "Furniture": [
        "sofa sectional couch living room furniture",
        "dining table chair set furniture ideas",
        "bookcase bookshelf storage unit furniture",
        "bed frame headboard platform furniture bedroom",
        "outdoor patio furniture set lounge chair",
        "affordable budget furniture home setup",
    ],
}

INTENT_PROTOTYPES = {
    "product-specific": [
        "best rug comparison review top rated buy",
        "affordable quality product recommendation purchase",
        "top 10 picks under budget review 2024",
        "where to buy product recommendation guide",
        "best air fryer comparison buying guide",
    ],
    "room-ideas": [
        "bedroom ideas inspiration transformation small space",
        "living room design aesthetic mood board ideas",
        "bathroom makeover before after renovation reveal",
        "cozy aesthetic room setup ideas",
    ],
    "outfit-style": [
        "outfit of the day styling inspiration look",
        "what to wear casual date night occasion",
        "aesthetic outfit ideas pinterest fashion",
    ],
    "food-recipe": [
        "easy recipe how to make step by step cooking",
        "dinner ideas quick recipe healthy meal prep",
        "recipe ingredients instructions method cooking",
    ],
    "food-baking": [
        "baking recipe from scratch tutorial beginner",
        "how to decorate cake cookie dessert frosting",
        "easy baking ideas project weekend",
    ],
    "diy-craft": [
        "DIY tutorial how to make craft project beginner",
        "step by step handmade homemade guide instructions",
        "easy weekend craft project activity",
    ],
    "hair-beauty": [
        "hair tutorial how to style at home easy",
        "makeup look tutorial step by step beginner",
        "skincare routine product recommendation review",
    ],
    "tattoo": [
        "tattoo ideas inspiration design gallery collection",
        "tattoo placement meaning symbolism ideas",
    ],
    "wedding-event": [
        "wedding inspiration planning ideas checklist",
        "party decoration theme setup ideas DIY budget",
    ],
    "pop-culture": [
        "disney theme party decoration merchandise",
        "fandom gift ideas merchandise themed room",
    ],
    "general": [
        "ideas tips guide information list collection",
    ],
}

STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","up","about","into","over","after","before","is","are","was",
    "were","be","been","have","has","do","does","did","will","would","could",
    "should","not","no","so","too","very","just","also","get","make","take",
    "your","my","our","their","this","that","these","those","what","which",
    "all","any","some","such","page","category","author","tag","post","blog",
    "www","com","net","org","admin",
}

YEAR_RE  = re.compile(r"\b(20\d{2}|19\d{2})\b")
SEP_RE   = re.compile(r"[-_]+")
ALPHA_RE = re.compile(r"[^a-z0-9 ]")

def clean_slug(path: str) -> str:
    segments = [s for s in path.split("/") if s and len(s) > 1]
    slug = max(segments, key=len) if segments else path
    slug = SEP_RE.sub(" ", slug)
    slug = YEAR_RE.sub("", slug)
    slug = ALPHA_RE.sub("", slug.lower()).strip()
    words = slug.split()
    while words and words[0] in STOP_WORDS: words.pop(0)
    while words and words[-1] in STOP_WORDS: words.pop()
    return " ".join(words) or slug

@st.cache_resource(show_spinner="Loading model all-MiniLM-L6-v2...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Building prototype embeddings...")
def build_matrices(_model):
    def centroid(protos):
        embs = _model.encode(protos, show_progress_bar=False, convert_to_numpy=True)
        return np.mean(embs, axis=0)
    n_labels = list(NICHE_PROTOTYPES.keys())
    n_matrix = np.stack([centroid(NICHE_PROTOTYPES[k]) for k in n_labels])
    i_labels = list(INTENT_PROTOTYPES.keys())
    i_matrix = np.stack([centroid(INTENT_PROTOTYPES[k]) for k in i_labels])
    return n_labels, n_matrix, i_labels, i_matrix

def classify(slugs, model, n_labels, n_matrix, i_labels, i_matrix, threshold=0.15):
    embs = model.encode(slugs, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    ns = cosine_similarity(embs, n_matrix)
    ins = cosine_similarity(embs, i_matrix)
    results = []
    for i in range(len(slugs)):
        ranked = np.argsort(ns[i])[::-1]
        pn = n_labels[ranked[0]]
        ps = float(ns[i][ranked[0]])
        sn = ""
        if len(ranked) > 1 and ps - float(ns[i][ranked[1]]) < threshold:
            sn = n_labels[ranked[1]]
        ti = int(np.argmax(ins[i]))
        results.append({
            "primary_niche":   pn,
            "secondary_niche": sn,
            "niche_score":     round(ps, 4),
            "intent":          i_labels[ti],
            "intent_score":    round(float(ins[i][ti]), 4),
        })
    return results

def aggregate_site_niches(df):
    records = []
    for site, grp in df.groupby("site"):
        scores = {}
        for _, row in grp.iterrows():
            w = float(row.get("amazon_clicks", 0) or 0) * 10 + float(row.get("sessions", 0) or 0) * 0.01
            w = max(w, 0.01)
            pn = row.get("primary_niche", "Other") or "Other"
            sn = row.get("secondary_niche", "") or ""
            scores[pn] = scores.get(pn, 0) + w
            if sn: scores[sn] = scores.get(sn, 0) + w * 0.35
        sorted_n = sorted(scores.items(), key=lambda x: -x[1])
        sn1 = sorted_n[0][0] if sorted_n else "Other"
        top = sorted_n[0][1] if sorted_n else 1.0
        sn2 = sorted_n[1][0] if len(sorted_n) > 1 and sorted_n[1][1] >= top * 0.20 else ""
        records.append({"site": site, "site_niche": sn1, "site_niche_secondary": sn2})
    return pd.DataFrame(records)

def parse_ga4_csv(text):
    if "---PATH_DATA---" in text:
        site_part, url_part = text.split("---PATH_DATA---", 1)
    else:
        site_part, url_part = "", text

    site_df = pd.DataFrame()
    if site_part.strip():
        site_df = pd.read_csv(StringIO(site_part.strip()))
        site_df.columns = [c.strip().lower().replace(" ", "_") for c in site_df.columns]

    url_part = url_part.strip()
    url_df = pd.read_csv(StringIO(url_part), skiprows=1)
    url_df.columns = ["site", "path", "sessions", "amazon_clicks", "avg_duration", "col6", "col7"]
    url_df = url_df.drop(columns=["col6", "col7"], errors="ignore")
    if "path" in url_df.columns:
        url_df = url_df[url_df["path"].notna() & url_df["path"].str.startswith("/", na=False)]
    for col in ["sessions", "amazon_clicks", "avg_duration"]:
        if col in url_df.columns:
            url_df[col] = pd.to_numeric(url_df[col], errors="coerce").fillna(0)
    return site_df, url_df.reset_index(drop=True)

# ── UI ───────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Settings")
    threshold = st.slider("Overlap threshold", 0.05, 0.30, 0.15, 0.01,
        help="Gap nhỏ hơn threshold → assign secondary niche")
    st.caption("Threshold càng nhỏ → ít overlap hơn")

with col2:
    st.subheader("📂 Upload CSV")
    uploaded = st.file_uploader("Upload GA4 export CSV", type=["csv"])

if uploaded:
    text = uploaded.read().decode("utf-8-sig")
    try:
        site_df, url_df = parse_ga4_csv(text)
    except Exception as e:
        st.error(f"❌ Parse error: {e}")
        st.stop()

    st.success(f"✅ Loaded: **{len(url_df)} URLs** từ **{url_df['site'].nunique()} sites**")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("URL rows", len(url_df))
    col_b.metric("Sites", url_df["site"].nunique())
    col_c.metric("Site rows", len(site_df))

    if st.button("🚀 Run Classification", type="primary", use_container_width=True):
        model = load_model()
        n_labels, n_matrix, i_labels, i_matrix = build_matrices(model)

        with st.spinner(f"Classifying {len(url_df)} URLs..."):
            slugs = url_df["path"].apply(clean_slug).tolist()
            results = classify(slugs, model, n_labels, n_matrix, i_labels, i_matrix, threshold)
            result_df = pd.DataFrame(results)
            url_df = pd.concat([url_df.reset_index(drop=True), result_df], axis=1)

        with st.spinner("Aggregating site niches..."):
            site_niches = aggregate_site_niches(url_df)
            url_df = url_df.merge(site_niches, on="site", how="left")

        st.success("✅ Classification done!")

        # Stats
        st.subheader("📊 Results")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Niche distribution (site_niche)**")
            st.dataframe(url_df["site_niche"].value_counts().reset_index()
                .rename(columns={"site_niche":"Niche","count":"URLs"}), use_container_width=True)
        with col2:
            st.write("**Intent distribution**")
            st.dataframe(url_df["intent"].value_counts().reset_index()
                .rename(columns={"intent":"Intent","count":"URLs"}), use_container_width=True)

        overlap = (url_df["site_niche_secondary"].notna() & (url_df["site_niche_secondary"] != "")).mean()
        st.info(f"🔁 Overlap rate: **{overlap*100:.1f}%** URLs có secondary niche")

        # Preview
        st.subheader("👀 Preview (50 rows)")
        st.dataframe(url_df[["site","path","sessions","amazon_clicks",
                              "primary_niche","intent","site_niche","site_niche_secondary"]].head(50),
                     use_container_width=True)

        # Export — giữ format gốc với ---PATH_DATA---
        out = StringIO()
        if not site_df.empty:
            site_df.to_csv(out, index=False)
        out.write("---PATH_DATA---\n")
        url_df.to_csv(out, index=False)
        out.seek(0)

        st.download_button(
            label="⬇️ Download ga4_enriched.csv",
            data=out.getvalue().encode("utf-8"),
            file_name="ga4_enriched.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )
else:
    st.info("👆 Upload file GA4 CSV để bắt đầu")
    with st.expander("📖 Format CSV yêu cầu"):
        st.code("""date,site,sessions,amazon_clicks,bounce_rate,avg_duration_sec,conversions
4/13/2026,example.com,1000,50,55.0,90,0
...
---PATH_DATA---
site,path,path_sessions,path_clicks,path_duration,,
example.com,/best-bedroom-rugs/,200,10,120,,
...""")