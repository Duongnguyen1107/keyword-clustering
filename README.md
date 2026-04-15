# Pinterest Affiliate — Keyword & URL Intelligence Tools

Bộ công cụ Python cho portfolio affiliate sites trên Pinterest.
Gồm 2 tools độc lập, dùng chung model `all-MiniLM-L6-v2`.

---

## Tool 1: Keyword Clustering (`app.py`)

Cluster keywords từ file Excel bằng semantic similarity.

### Chạy
```bash
streamlit run app.py
```

### Input
File `.xlsx` với cột `keyword`.

### Parameters
| Param | Default | Ý nghĩa |
|---|---|---|
| `--threshold` | 0.82 | Cluster similarity cutoff |
| `--sim-filter` | 0.88 | Pair filter threshold |
| `--max-size` | 20 | Max keywords per cluster |

---

## Tool 2: URL Semantic Classifier (`classify_urls.py`)

Classify URL slugs từ GA4 export → niche + intent bằng semantic embeddings.
Thêm 7 columns vào CSV: `primary_niche`, `secondary_niche`, `niche_score`,
`intent`, `intent_score`, `site_niche`, `site_niche_secondary`.

### Chạy
```bash
python classify_urls.py --input ga4_raw.csv --output ga4_enriched.csv
```

### Options
| Option | Default | Ý nghĩa |
|---|---|---|
| `--input` | — | Input GA4 CSV |
| `--output` | — | Output enriched CSV |
| `--model` | all-MiniLM-L6-v2 | Sentence transformer model |
| `--threshold` | 0.15 | Overlap gap threshold |
| `--batch-size` | 64 | Encoding batch size |

### Niches được classify
`Home Decor`, `Garden/Outdoor`, `Kitchen`, `Food/Recipe`, `Food/Baking`,
`Styling`, `Hair/Beauty`, `Tattoo`, `Wedding/Event`, `DIY/Craft`,
`Lifestyle`, `Furniture`

### Intents được classify
`product-specific`, `room-ideas`, `outfit-style`, `food-recipe`,
`food-baking`, `diy-craft`, `hair-beauty`, `tattoo`, `wedding-event`,
`pop-culture`, `general`

---

## Setup

```bash
# Python 3.11
pip install -r requirements.txt --break-system-packages

# Lần đầu — model sẽ tự download (~90MB)
python classify_urls.py --input sample.csv --output out.csv
```

---

## Workflow hàng ngày

```
GA4 Apps Script export → ga4_raw.csv
         ↓
python classify_urls.py --input ga4_raw.csv --output ga4_enriched.csv
         ↓  (~2-3 phút cho 3000 URLs)
Upload ga4_enriched.csv lên GA4 Dashboard (HTML)
```