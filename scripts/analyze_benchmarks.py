import pandas as pd
import numpy as np

f1 = 'data/outputs/benchmark_audit_20261114_024442.xlsx'
f2 = 'data/outputs/benchmark_audit_20261115_162516.xlsx'

df1 = pd.read_excel(f1, sheet_name='Audit Details')
df2 = pd.read_excel(f2, sheet_name='Audit Details')

print("=" * 80)
print("BENCHMARK COMPARISON SUMMARY")
print("=" * 80)
print(f"File 1 (2026-11-14): {len(df1)} samples")
print(f"File 2 (2026-11-15): {len(df2)} samples")

print("\n--- CONFUSION MATRIX BREAKDOWN ---")
print("File 1 (2026-11-14):")
print(df1['classification_type'].value_counts())
print("\nFile 2 (2026-11-15):")
print(df2['classification_type'].value_counts())

# Check overlap of queries
q1 = set(df1['query'])
q2 = set(df2['query'])
common_queries = q1.intersection(q2)
print(f"\nCommon queries between the two runs: {len(common_queries)}")

# Merge on query for direct comparison
df_merged = pd.merge(
    df1[['query', 'ground_truth', 'prediction', 'classification_type', 'ai_model_verdict', 'consensus_verdict', 'verdict_label', 'confidence_score', 'reasons', 'source_url']],
    df2[['query', 'ground_truth', 'prediction', 'classification_type', 'ai_model_verdict', 'consensus_verdict', 'verdict_label', 'confidence_score', 'reasons', 'source_url']],
    on=['query', 'ground_truth'],
    suffixes=('_20261114', '_20261115')
)
print(f"Merged exact matches: {len(df_merged)}")

# Shift in classification types
print("\n--- SHIFT MATRIX (2026-11-14 -> 2026-11-15) ---")
shift = pd.crosstab(df_merged['classification_type_20261114'], df_merged['classification_type_20261115'])
print(shift)

# Look specifically at tracks that were TP on 14th but became FN on 15th
tp_to_fn = df_merged[(df_merged['classification_type_20261114'].str.startswith('TP')) & (df_merged['classification_type_20261115'].str.startswith('FN'))]
print(f"\nTracks that were TP in 20261114 but became FN in 20261115: {len(tp_to_fn)}")

print("\nSample of TP -> FN tracks:")
for i, (_, row) in enumerate(tp_to_fn.head(15).iterrows()):
    print(f"\n[{i+1}] {row['query']}")
    print(f"   20261114: Pred={row['prediction_20261114']} | Model={row['ai_model_verdict_20261114']} | Consensus={row['consensus_verdict_20261114']} | Conf={row['confidence_score_20261114']}")
    print(f"   20261115: Pred={row['prediction_20261115']} | Model={row['ai_model_verdict_20261115']} | Consensus={row['consensus_verdict_20261115']} | Conf={row['confidence_score_20261115']}")
    r1 = str(row['reasons_20261114'])[:140]
    r2 = str(row['reasons_20261115'])[:140]
    print(f"   Reason 14: {r1}")
    print(f"   Reason 15: {r2}")
