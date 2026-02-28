# CMeKG 数据集获取与使用指南

## 一、数据集下载

### 方式1：阿里云天池（推荐）

1. 访问：https://tianchi.aliyun.com/dataset/81506
2. 注册/登录阿里云账号
3. 点击"下载"获取数据集
4. 数据集通常包含：
   - CMeKG_triples.csv（三元组数据）
   - 或其他格式的关系数据

### 方式2：官方申请

1. 访问：http://cmekg.pcl.ac.cn/
2. 填写申请表（学术研究用途）
3. 等待审核通过
4. 下载完整数据集

### 方式3：开放知识图谱

1. 访问：http://openkg.cn/
2. 搜索"医学"或"医疗"
3. 查找可用的知识图谱数据

---

## 二、数据格式说明

### 三元组格式 (head, relation, tail)

```
疾病, 症状, 发热
感冒, 治疗方法, 阿司匹林
高血压, 病因, 遗传因素
```

---

## 三、导入到项目

下载完成后，将数据集放到项目目录：

```bash
# 假设下载的文件是 CMeKG_triples.csv
cp CMeKG_triples.csv backend/
cd backend
python import_data.py
```

---

## 四、注意事项

- CMeKG 数据集仅限**学术研究**使用
- 请勿用于商业用途
- 数据引用请注明来源

---

## 五、替代方案

如果无法获取 CMeKG，可以考虑：

1. **继续使用当前数据集**（251实体，已验证可用）
2. **自行构建小型医疗知识图谱**
3. **使用其他开放医疗数据集**

## 六、联系方式

- CMeKG 官方：http://cmekg.pcl.ac.cn/
- 阿里云天池：https://tianchi.aliyun.com/dataset/81506
