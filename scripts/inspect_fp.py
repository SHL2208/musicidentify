import pandas as pd

f1 = 'data/outputs/benchmark_audit_20261114_024442.xlsx'
f2 = 'data/outputs/benchmark_audit_20261115_162516.xlsx'

df1 = pd.read_excel(f1, sheet_name='Audit Details')
df2 = pd.read_excel(f2, sheet_name='Audit Details')

fp1 = df1[df1['classification_type'].str.contains('FP', na=False)]
fp2 = df2[df2['classification_type'].str.contains('FP', na=False)]

print(f"Total FP in 2026-11-14: {len(fp1)}")
print(f"Total FP in 2026-11-15: {len(fp2)}")

print("\n--- FP in 2026-11-15 (Vu oan nhạc người) ---")
for idx, r in fp2.iterrows():
    print(f"Track: {r['query']} | Pred: {r['prediction']} | ModelVerdict: {r['ai_model_verdict']} | Consensus: {r['consensus_verdict']}")
    print(f"   Reason: {str(r['reasons'])[:150]}...")
    print(f"   URL: {r['source_url']}")

print("\n--- FP in 2026-11-14 that became TN in 2026-11-15 (Fixed FP) ---")
fp1_queries = set(fp1['query'])
fp2_queries = set(fp2['query'])
fixed_fp = fp1_queries - fp2_queries
print(f"Fixed FP count: {len(fixed_fp)}")
for q in list(fixed_fp)[:10]:
    row1 = fp1[fp1['query'] == q].iloc[0]
    row2 = df2[df2['query'] == q].iloc[0] if q in set(df2['query']) else None
    print(f"- {q}")
    print(f"   14: {row1['classification_type']} -> Reason: {str(row1['reasons'])[:100]}")
    if row2 is not None:
        print(f"   15: {row2['classification_type']} -> Reason: {str(row2['reasons'])[:100]}")
