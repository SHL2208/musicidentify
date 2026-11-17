import pandas as pd

df = pd.read_excel('testcase4.xlsx')
titles = ['Cố Quên Người Đổi Thay', 'Yêu Không Toan Tính 3', 'Tiếng Yêu Đầu Vỡ Đôi 2', 'Buông Tay', 'Tàn Nhẫn', 'Nhạc Tàn Người Cũng Tan', 'Hạt Bụi Nhỏ', 'Ơn Tổ Nghề']

for t in titles:
    sub = df[df['title'].astype(str).str.contains(t, case=False, na=False)]
    for idx, r in sub.iterrows():
        print(f"Row {idx} | {r['title']} | {r['artists']} | AI YES: {r['AI YES']} | NOTE: {r['NOTE']}")
