import pandas as pd

df = pd.read_excel('testcase4.xlsx')
print('testcase4 shape:', df.shape)
print('AI YES value counts:')
print(df['AI YES'].value_counts(dropna=False))

sample_queries = ['Tuổi Nàng 15', '36 KINGDOM', 'DK Media', 'Bayanilak', 'Kaiba Shinsui', 'Lil Pang Wag']
for sq in sample_queries:
    sub = df[df['title'].astype(str).str.contains(sq, case=False, na=False) | df['artists'].astype(str).str.contains(sq, case=False, na=False)]
    print(f"\nSearch for '{sq}': {len(sub)} matches")
    for idx, r in sub.head(3).iterrows():
        print(f"   {r['title']} | {r['artists']} | Song Type: {r['Song Type']} | AI YES: {r['AI YES']} | NOTE: {r['NOTE']}")
