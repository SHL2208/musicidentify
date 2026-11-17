import pandas as pd

df = pd.read_excel('testcase4.xlsx')

test_names = ['2Can', 'Chí Thành Danh', 'Bảo Thạch', 'Lâm Chấn Kiệt', 'Vương Lâm', 'Ivan Tran', 'Neit', 'Trixie Jane', 'Arlo Fontaine']

print("INSPECTING GROUND TRUTH IN testcase4.xlsx FOR SUSPICIOUS AI LABELS:")
for name in test_names:
    sub = df[df['artists'].astype(str).str.contains(name, case=False, na=False) | df['title'].astype(str).str.contains(name, case=False, na=False)]
    print(f"\nSearch for '{name}': {len(sub)} rows")
    for idx, r in sub.head(4).iterrows():
        print(f"   Row {idx} | {r['title']} | {r['artists']} | Song Type: {r['Song Type']} | AI YES: {r['AI YES']} | NOTE: {r['NOTE']}")
