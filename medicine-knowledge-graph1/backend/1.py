import pandas as pd

# 读取 CSV 文件，使用制表符 \t 作为分隔符
df = pd.read_csv(r"D:\Desktop\文件\课程文件\人工知识图谱\医疗知识问答系统\medicine-knowledge-graph1\backend\ownthink_v2.csv", sep='\t')

# 输出前 5 行数据
print("前 5 行数据：")
print(df.head())

# 输出列名
print("\n列名：")
print(df.columns)

# 输出空值统计
print("\n空值统计：")
print(df.isna().sum())

# 输出前 50 osi
print("\n前 50 行数据：")
print(df.head(50))