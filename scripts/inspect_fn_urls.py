import pandas as pd

f2 = 'data/outputs/benchmark_audit_20261115_162516.xlsx'
df2 = pd.read_excel(f2, sheet_name='Audit Details')

fn_non_cat = df2[
    (df2['classification_type'].str.contains('FN', na=False)) &
    (~df2['artist'].str.contains('Picture Record|Bayanilak|Toymiet Sun|Lil Pang Wag|Ashe|Mokai Tunr|Kaiba Shinsui|Tokai Yussie|DJ JIN JIN|Lucid Analine|Caitlyn Lynn|Samueluna|Dame Wing Hear|Dark Suiltion|YoungYota|Mikasa Owen|Krisbone Crow|Roman Donovan', case=False, na=False))
]

print(f"Non-catalog FN count: {len(fn_non_cat)}")
for idx, r in fn_non_cat.iterrows():
    print(f"\n[{idx}] {r['query']}")
    print(f"   Match Score: {r['match_score']} | Verified: {r['is_verified_source']}")
    print(f"   URL: {r['source_url']}")
    print(f"   Reason: {str(r['reasons'])[:120]}...")
