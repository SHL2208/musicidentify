import pandas as pd
import json

f2 = 'data/outputs/benchmark_audit_20261115_162516.xlsx'
df2 = pd.read_excel(f2, sheet_name='Audit Details')

# Filter FN
fn2 = df2[df2['classification_type'].str.contains('FN', na=False)]
print(f"Total FN in 2026-11-15: {len(fn2)}")

with open('data/catalogs/catalog_snapshot.json', 'r', encoding='utf-8') as f:
    cat = json.load(f)

ai_artists = cat.get('ai_artists', [])
ai_distributors = cat.get('ai_distributors', [])

# Check reasons and categories of FN
categories = {
    'ai_artist_in_catalog': [],
    'spotify_early_exit': [],
    'gemini_hallucination_or_acoustic': [],
    'other': []
}

for idx, row in fn2.iterrows():
    q = str(row['query'])
    artist = str(row['artist'])
    title = str(row['song_title'])
    reasons = str(row['reasons'])
    verdict = str(row['consensus_verdict'])
    source_url = str(row['source_url'])
    
    # Check if artist is in ai_artists
    in_ai_cat = any(artist.lower() in str(x).lower() for x in ai_artists)
    
    if "Official release in" in reasons or "trước kỷ nguyên nhạc AI" in reasons or "spotify-pre-ai-rule" in str(row.get('model_used', '')):
        categories['spotify_early_exit'].append((q, artist, reasons[:120]))
    elif in_ai_cat:
        categories['ai_artist_in_catalog'].append((q, artist, reasons[:120]))
    else:
        categories['gemini_hallucination_or_acoustic'].append((q, artist, reasons[:120]))

print(f"\nBreakdown of 56 FN cases in 2026-11-15:")
print(f"1. AI Artist is in Catalog (but was NOT overridden or audio evidence threshold failed): {len(categories['ai_artist_in_catalog'])}")
print(f"2. Spotify Pre-AI Early Exit (wrong release date / false match to pre-2022 human track): {len(categories['spotify_early_exit'])}")
print(f"3. General Acoustic / Gemini Miss (Not in catalog, missed by both): {len(categories['gemini_hallucination_or_acoustic'])}")

print("\n--- Details of AI Artist in Catalog missed ---")
for q, a, r in categories['ai_artist_in_catalog'][:15]:
    print(f"- {q} (Artist: {a})")

print("\n--- Details of Spotify Early Exit missed ---")
for q, a, r in categories['spotify_early_exit']:
    print(f"- {q} (Artist: {a}) -> {r}")

print("\n--- Details of General Acoustic / Gemini Miss ---")
for q, a, r in categories['gemini_hallucination_or_acoustic'][:15]:
    print(f"- {q} (Artist: {a}) -> {r}")
