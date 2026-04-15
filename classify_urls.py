#!/usr/bin/env python3
"""
classify_urls.py — Semantic URL Classifier for GA4 Dashboard
=====================================================================
Dùng sentence-transformers để classify URL slugs → niche + intent
bằng cosine similarity với prototype embeddings.

Usage:
    python classify_urls.py --input ga4_raw.csv --output ga4_enriched.csv

Output thêm các columns vào CSV:
    primary_niche, secondary_niche, niche_score,
    intent, intent_score, site_niche, site_niche_secondary
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Prototype definitions ────────────────────────────────────────────────────
# Viết như natural language sentences — model hiểu ngữ nghĩa, không phải keywords

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
        "greenhouse herb garden growing seeds",
        "outdoor string lights patio ambiance",
        "garden tools planting watering hose",
        "pergola deck fence outdoor structure",
        "bird feeder wildlife nature garden",
    ],
    "Kitchen": [
        "kitchen organization storage solutions countertop",
        "cookware pan pot set kitchen tools",
        "coffee maker espresso machine kitchen appliance",
        "knife set cutting board kitchen gadget",
        "kitchen shelf cabinet pantry organizer",
        "air fryer instant pot slow cooker pressure cooker",
        "kitchen decor aesthetic farmhouse modern",
        "dish rack utensil holder kitchen accessories",
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
        "meal prep batch cooking weekly plan",
        "comfort food hearty filling dinner recipe",
    ],
    "Food/Baking": [
        "chocolate cake cupcake dessert recipe from scratch",
        "cookie brownie bar baking easy beginner",
        "sourdough bread muffin scone pastry baking",
        "frosting buttercream icing fondant cake decorating",
        "cheesecake no bake dessert easy recipe",
        "birthday cake ideas decoration tutorial",
        "holiday Christmas Easter baking treats",
        "pie tart galette pastry shell filling",
    ],
    "Styling": [
        "outfit ideas what to wear casual everyday",
        "fashion style clothing aesthetic look inspiration",
        "dress jeans boots sneakers shoes styling",
        "capsule wardrobe minimalist fashion basics",
        "summer winter fall spring seasonal outfit",
        "bag purse handbag accessory jewelry styling",
        "body type flattering clothes styling tips",
        "street style trendy fashion look",
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
        "hair care treatment mask growth tips",
        "perfume fragrance beauty recommendation",
    ],
    "Tattoo": [
        "tattoo design ideas inspiration placement",
        "small fine line minimalist tattoo art",
        "sleeve floral geometric mandala tattoo",
        "meaningful symbol quote tattoo ideas",
        "tattoo aftercare healing moisturizer",
        "watercolor blackwork traditional tattoo style",
        "feminine delicate tattoo ideas for women",
    ],
    "Wedding/Event": [
        "wedding decoration ceremony reception ideas",
        "bridal shower bachelorette party ideas themes",
        "engagement proposal anniversary romantic ideas",
        "baby shower gender reveal party decoration",
        "birthday party table decoration theme setup",
        "wedding floral arrangement centerpiece bouquet",
        "DIY wedding craft decoration handmade budget",
        "wedding dress bridesmaid gown style",
        "wedding favor gift guest table seating",
    ],
    "DIY/Craft": [
        "DIY home project tutorial step by step beginner",
        "crochet knitting sewing pattern handmade craft",
        "woodworking build shelf furniture project",
        "painting canvas art craft kids activity",
        "repurpose upcycle thrift flip makeover project",
        "resin pour clay pottery craft tutorial",
        "macrame wreath candle making craft project",
    ],
    "Lifestyle": [
        "morning routine productivity self care habits",
        "travel destination guide bucket list tips",
        "minimalist lifestyle wellness mental health",
        "cozy reading book recommendation hobby",
        "personal finance budget saving money tips",
        "journal planner goal setting motivation",
        "digital nomad remote work lifestyle",
    ],
    "Furniture": [
        "sofa sectional couch living room furniture",
        "dining table chair set furniture ideas",
        "bookcase bookshelf storage unit furniture",
        "bed frame headboard platform furniture bedroom",
        "outdoor patio furniture set lounge chair",
        "accent furniture side table console entry",
        "furniture makeover paint chalk restoration",
        "affordable budget furniture home setup",
    ],
}

INTENT_PROTOTYPES = {
    "product-specific": [
        "best rug comparison review top rated buy",
        "affordable quality product recommendation purchase",
        "top 10 picks under budget review 2024",
        "where to buy product recommendation guide",
        "product review honest opinion worth it",
        "best air fryer comparison buying guide",
        "cheap affordable budget option recommendation",
    ],
    "room-ideas": [
        "bedroom ideas inspiration transformation small space",
        "living room design aesthetic mood board ideas",
        "bathroom makeover before after renovation reveal",
        "apartment rental decor ideas no damage",
        "home tour interior design room inspiration",
        "cozy aesthetic room setup ideas",
    ],
    "outfit-style": [
        "outfit of the day styling inspiration look",
        "what to wear casual date night occasion",
        "how to style layering mixing matching outfit",
        "aesthetic outfit ideas pinterest fashion",
        "complete look head to toe styling",
    ],
    "food-recipe": [
        "easy recipe how to make step by step cooking",
        "dinner ideas quick recipe for the week",
        "healthy meal prep recipe collection batch",
        "recipe ingredients instructions method cooking",
    ],
    "food-baking": [
        "baking recipe from scratch tutorial beginner",
        "how to decorate cake cookie dessert frosting",
        "easy baking ideas project weekend",
        "baking tips techniques tricks tutorial",
    ],
    "diy-craft": [
        "DIY tutorial how to make craft project beginner",
        "step by step handmade homemade guide instructions",
        "easy weekend craft project activity",
        "make your own build tutorial materials needed",
    ],
    "hair-beauty": [
        "hair tutorial how to style at home easy",
        "makeup look tutorial step by step beginner",
        "skincare routine product recommendation review",
        "hair transformation before after color cut",
    ],
    "tattoo": [
        "tattoo ideas inspiration design gallery collection",
        "tattoo placement meaning symbolism ideas",
        "tattoo style guide what to choose",
    ],
    "wedding-event": [
        "wedding inspiration planning ideas checklist",
        "party decoration theme setup ideas DIY budget",
        "event planning tips ideas inspiration",
    ],
    "pop-culture": [
        "disney theme party decoration merchandise",
        "fandom gift ideas merchandise themed",
        "movie show character themed decor room",
    ],
    "general": [
        "ideas tips guide information list collection",
        "how to tips advice beginner guide",
    ],
}

# ── Stop words — bỏ khi clean slug ─────────────────────────────────────────
STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","up","about","into","over","after","before","is","are","was",
    "were","be","been","have","has","do","does","did","will","would","could",
    "should","not","no","so","too","very","just","also","get","make","take",
    "your","my","our","their","this","that","these","those","what","which",
    "all","any","some","such","page","category","author","tag","post","blog",
    "www","com","net","org","admin",
}

YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
SEP_RE = re.compile(r"[-_]+")
NONALPHA_RE = re.compile(r"[^a-z0-9 ]")


def clean_slug(path: str) -> str:
    """Extract meaningful text từ URL path."""
    segments = [s for s in path.split("/") if s and len(s) > 1]
    # Bỏ segments ngắn (category, tags) — lấy segment dài nhất
    slug = max(segments, key=len) if segments else path
    slug = SEP_RE.sub(" ", slug)
    slug = YEAR_RE.sub("", slug)
    slug = NONALPHA_RE.sub("", slug.lower()).strip()
    # Strip stop words ở đầu/cuối, giữ ở giữa
    words = slug.split()
    while words and words[0] in STOP_WORDS:
        words.pop(0)
    while words and words[-1] in STOP_WORDS:
        words.pop()
    return " ".join(words) or slug


# ── Model loading ────────────────────────────────────────────────────────────

def build_prototype_matrix(model: SentenceTransformer, prototypes: dict) -> tuple:
    """Encode tất cả prototypes → centroid embeddings."""
    labels = list(prototypes.keys())
    centroids = []
    for label in labels:
        embs = model.encode(
            prototypes[label],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        centroids.append(np.mean(embs, axis=0))
    matrix = np.stack(centroids)  # (n_labels, dim)
    return labels, matrix


# ── Classification core ──────────────────────────────────────────────────────

def classify_slugs(
    slugs: list,
    model: SentenceTransformer,
    niche_labels: list,
    niche_matrix: np.ndarray,
    intent_labels: list,
    intent_matrix: np.ndarray,
    overlap_threshold: float = 0.15,
    batch_size: int = 64,
) -> list:
    """
    Classify list of slug strings.
    overlap_threshold: nếu top-2 niche score cách nhau < threshold → assign cả 2.
    """
    if not slugs:
        return []

    # Encode batch
    slug_embs = model.encode(
        slugs,
        batch_size=batch_size,
        show_progress_bar=len(slugs) > 200,
        convert_to_numpy=True,
    )

    niche_sims = cosine_similarity(slug_embs, niche_matrix)   # (n, n_niches)
    intent_sims = cosine_similarity(slug_embs, intent_matrix)  # (n, n_intents)

    results = []
    for i in range(len(slugs)):
        # ── Niche ──
        ns = niche_sims[i]
        ranked = np.argsort(ns)[::-1]
        p_niche = niche_labels[ranked[0]]
        p_score = float(ns[ranked[0]])

        s_niche = ""
        if len(ranked) > 1:
            s_score = float(ns[ranked[1]])
            if p_score - s_score < overlap_threshold:
                s_niche = niche_labels[ranked[1]]

        # ── Intent ──
        ins = intent_sims[i]
        top_i = int(np.argmax(ins))
        intent = intent_labels[top_i]
        i_score = float(ins[top_i])

        results.append(
            {
                "primary_niche": p_niche,
                "secondary_niche": s_niche,
                "niche_score": round(p_score, 4),
                "intent": intent,
                "intent_score": round(i_score, 4),
            }
        )
    return results


# ── Site-level aggregation ───────────────────────────────────────────────────

def aggregate_site_niches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate URL-level niches → site-level dominant niche.
    Weight: clicks × 10 + sessions × 0.01 (clicks-first).
    """
    records = []
    for site, grp in df.groupby("site"):
        scores: dict = {}
        for _, row in grp.iterrows():
            clicks = float(row.get("amazon_clicks", 0) or 0)
            sess = float(row.get("sessions", 0) or 0)
            weight = clicks * 10 + sess * 0.01
            if weight < 0.01:
                weight = 0.01  # floor — jangan ignore zero-click pages

            pn = row.get("primary_niche", "Other") or "Other"
            sn = row.get("secondary_niche", "") or ""

            scores[pn] = scores.get(pn, 0) + weight
            if sn:
                scores[sn] = scores.get(sn, 0) + weight * 0.35

        sorted_n = sorted(scores.items(), key=lambda x: -x[1])
        site_niche = sorted_n[0][0] if sorted_n else "Other"
        top_score = sorted_n[0][1] if sorted_n else 1.0

        site_niche2 = ""
        if len(sorted_n) > 1 and sorted_n[1][1] >= top_score * 0.20:
            site_niche2 = sorted_n[1][0]

        records.append(
            {
                "site": site,
                "site_niche": site_niche,
                "site_niche_secondary": site_niche2,
            }
        )
    return pd.DataFrame(records)


# ── CSV parser — handle both site-level và URL-level rows ───────────────────

def parse_ga4_csv(path: str) -> tuple:
    """
    Parse GA4 export CSV.
    Returns (site_df, url_df) — cả hai có thể None nếu không detect được.
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Detect URL rows: có cột 'path' hoặc column thứ 2 là path-like
    if "path" in df.columns:
        url_df = df[df["path"].notna() & df["path"].str.startswith("/")].copy()
        site_df = df[~df["path"].str.startswith("/", na=False)].copy()
    else:
        # Fallback: cột index 1 là path
        cols = list(df.columns)
        if len(cols) >= 2:
            df = df.rename(columns={cols[1]: "path"})
            url_df = df[df["path"].notna() & df["path"].str.startswith("/")].copy()
            site_df = df[~df["path"].str.startswith("/", na=False)].copy()
        else:
            url_df = df.copy()
            site_df = pd.DataFrame()

    # Normalize numeric columns
    for col in ["sessions", "amazon_clicks", "avg_duration", "bounce_rate"]:
        if col in url_df.columns:
            url_df[col] = pd.to_numeric(url_df[col], errors="coerce").fillna(0)

    return site_df, url_df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Semantic URL classifier — adds niche + intent columns to GA4 CSV"
    )
    parser.add_argument("--input",     required=True,  help="Input GA4 CSV path")
    parser.add_argument("--output",    required=True,  help="Output enriched CSV path")
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Niche overlap threshold: gap < threshold → assign secondary niche (default: 0.15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Encoding batch size (default: 64)",
    )
    args = parser.parse_args()

    # ── Load CSV ──
    print(f"📂 Reading: {args.input}")
    try:
        site_df, url_df = parse_ga4_csv(args.input)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        sys.exit(1)

    if url_df.empty:
        print("❌ No URL rows found. Check CSV format.")
        sys.exit(1)

    print(f"   URL rows: {len(url_df)}")
    if "site" in url_df.columns:
        print(f"   Sites: {url_df['site'].nunique()}")

    # ── Load model ──
    print(f"\n🤖 Loading model: {args.model}")
    try:
        model = SentenceTransformer(args.model)
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        sys.exit(1)

    # ── Build prototype matrices ──
    print("📐 Building prototype embeddings...")
    niche_labels, niche_matrix = build_prototype_matrix(model, NICHE_PROTOTYPES)
    intent_labels, intent_matrix = build_prototype_matrix(model, INTENT_PROTOTYPES)
    print(f"   Niches: {len(niche_labels)}")
    print(f"   Intents: {len(intent_labels)}")

    # ── Clean slugs ──
    print("\n🔤 Cleaning URL slugs...")
    url_df = url_df.reset_index(drop=True)
    slugs = url_df["path"].apply(clean_slug).tolist()

    # ── Classify ──
    print(f"🔍 Classifying {len(slugs)} URLs...")
    results = classify_slugs(
        slugs,
        model,
        niche_labels,
        niche_matrix,
        intent_labels,
        intent_matrix,
        overlap_threshold=args.threshold,
        batch_size=args.batch_size,
    )
    result_df = pd.DataFrame(results)
    url_df = pd.concat([url_df, result_df], axis=1)

    # ── Site-level aggregation ──
    if "site" in url_df.columns:
        print("🏢 Aggregating site-level niches...")
        site_niches = aggregate_site_niches(url_df)
        url_df = url_df.merge(site_niches, on="site", how="left")
    else:
        print("⚠️  No 'site' column — skipping site-level aggregation")
        url_df["site_niche"] = url_df["primary_niche"]
        url_df["site_niche_secondary"] = url_df["secondary_niche"]

    # ── Save ──
    url_df.to_csv(args.output, index=False)
    print(f"\n✅ Saved: {args.output}")
    print(f"   {len(url_df)} URLs classified\n")

    # Summary
    print("📊 Niche distribution (site_niche):")
    print(url_df["site_niche"].value_counts().to_string())
    print("\n🎯 Intent distribution:")
    print(url_df["intent"].value_counts().to_string())
    print("\n🔁 Overlap rate (sites với secondary niche):")
    overlap = url_df["site_niche_secondary"].notna() & (url_df["site_niche_secondary"] != "")
    print(f"   {overlap.sum()} / {len(url_df)} URLs ({overlap.mean()*100:.1f}%)")


if __name__ == "__main__":
    main()