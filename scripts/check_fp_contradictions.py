import pandas as pd

f2 = 'data/outputs/benchmark_audit_20261115_162516.xlsx'
df2 = pd.read_excel(f2, sheet_name='Audit Details')
fp2 = df2[df2['classification_type'].str.contains('FP', na=False)]

print("ALL 10 FALSE POSITIVES IN 2026-11-15:")
for idx, r in fp2.iterrows():
    print(f"\n--- [{idx}] {r['song_title']} - {r['artist']} ---")
    print(f"Media ID: {r['media_id']} | Query: {r['query']}")
    print(f"GT: {r['ground_truth']} | Pred: {r['prediction']} | Conf: {r['confidence_score']}")
    print(f"Consensus Verdict: {r['consensus_verdict']} | Model Verdict: {r['ai_model_verdict']}")
    print(f"URL: {r['source_url']}")
    print(f"Reasons: {r['reasons']}")
