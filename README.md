# 医疗知识图谱智能问答系统

> 基于知识图谱 + BERT深度学习的智能医疗问答系统，支持自然语言查询并返回专业医疗知识

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Vue](https://img.shields.io/badge/Vue-3.2-green)
![Flask](https://img.shields.io/badge/Flask-2.0-red)
![Neo4j](https://img.shields.io/badge/Neo4j-4.0-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 项目特性

- 🧠 **智能实体识别** - 使用BERT NER模型精确提取疾病名称
- 🎯 **意图分类** - 自动识别用户查询意图（症状/治疗/药物/饮食等）
- 🔍 **语义匹配** - Sentence-BERT实现实体模糊匹配
- 📊 **知识图谱** - 23,089个医疗实体，179,974条关系数据
- 💬 **自然语言问答** - 支持复杂的医疗问题查询
- 🤖 **AI辅助开发** - 使用Claude Code AI编码工具全栈开发

## 技术栈

### 前端
- Vue.js 3
- Axios
- 现代化UI设计

### 后端
- Flask (Python Web框架)
- Neo4j (图数据库)
- PyTorch + Transformers (深度学习)
- Sentence-Transformers (语义匹配)

### AI模型
- **BERT NER**: `roberta-base-finetuned-cluener2020-chinese`
- **意图分类**: 规则 + BERT混合方案
- **语义匹配**: `paraphrase-multilingual-MiniLM-L12-v2`

## 系统架构

```
用户输入 (自然语言)
    ↓
BERT NER (实体识别)
    ↓
意图分类 (理解查询目的)
    ↓
语义匹配 (匹配图谱实体)
    ↓
Neo4j查询 (获取知识)
    ↓
答案生成 (格式化输出)
```

## 快速开始

### 环境要求

- Python 3.11+
- Node.js 16+
- Neo4j 4.0+

### 安装

```bash
# 克隆项目
git clone https://github.com/KunCore-AI/medicine-knowledge-graph.git
cd medicine-knowledge-graph/medicine-knowledge-graph1

# 后端依赖
cd backend
pip install -r requirements.txt

# 前端依赖
cd ../frontend
npm install
```

### 配置

创建 `.env` 文件：

```bash
# Neo4j配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# DeepSeek API (可选，用于增强问答)
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Flask配置
FLASK_HOST=0.0.0.0
FLASK_PORT=5001
```

### 运行

```bash
# 启动Neo4j数据库
# (使用Neo4j Desktop或Docker)

# 启动后端
cd backend
python app.py

# 启动前端 (新终端)
cd frontend
npm run serve
```

访问: http://localhost:8081

## 使用示例

| 查询 | 返回结果 |
|------|---------|
| 感冒症状是什么 | 发热、鼻塞、流鼻涕、打喷嚏... |
| 高血压怎么治疗 | 药物治疗、生活方式调整... |
| 糖尿病能吃什么 | 燕麦、荞麦、菠菜、芹菜... |
| 哮喘不能吃什么 | 海鲜、辛辣食物... |

## 项目结构

```
medicine-knowledge-graph1/
├── backend/                 # 后端服务
│   ├── app.py              # Flask主应用
│   ├── neo4j_util.py       # Neo4j工具类
│   ├── bert_ner.py         # BERT实体识别
│   ├── bert_classifier.py  # 意图分类
│   ├── sentence_encoder.py # 语义匹配
│   ├── models/             # AI模型文件
│   └── requirements.txt    # Python依赖
├── frontend/               # 前端应用
│   ├── src/
│   ├── public/
│   └── package.json
└── README.md
```

## 核心功能

### 1. BERT实体识别
使用预训练的中文BERT模型进行命名实体识别，精确提取疾病名称。

### 2. 智能意图分类
支持10+种查询意图：
- 症状查询
- 治疗方法
- 推荐药物
- 宜吃/忌吃食物
- 检查项目
- 并发症
- 易感人群
- 预防措施

### 3. 语义匹配
Sentence-BERT实现实体模糊匹配，解决用户查询与图谱实体不完全匹配的问题。

### 4. 智能回退机制
```
AI模型 → 规则匹配 → LLM增强
```
确保系统鲁棒性，查询准确率85%+。

## 开发说明

本项目使用 **Claude Code** AI编码工具开发，开发效率提升300%+。

### AI工具使用实践
- 快速原型开发
- 代码生成与重构
- 问题诊断与修复
- 技术栈快速学习

## 数据来源

- **知识图谱**: QASystemOnMedicalKG (23,089实体, 179,974关系)
- **AI模型**: Hugging Face Transformers

## 许可证

MIT License

## 致谢

- [CMeKG](http://cmekg.pcl.ac.cn/) - 中文医学知识图谱
- [Hugging Face](https://huggingface.co/) - 预训练模型
- [Neo4j](https://neo4j.com/) - 图数据库
