# 医疗知识问答系统

基于 Neo4j 图数据库和 DeepSeek API 的智能医疗知识问答系统。

## 功能特性

- 🩺 **知识图谱查询**: 基于 Neo4j 存储 251+ 医疗实体、277+ 关系
- 🤖 **AI 辅助回答**: 集成 DeepSeek API 处理知识图谱无法回答的问题
- 💬 **对话式界面**: 简洁友好的 Web 问答界面
- 🔒 **安全配置**: 环境变量管理敏感信息

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端 | Python + Flask |
| 数据库 | Neo4j 图数据库 |
| 前端 | Vue 3 + Axios |
| NLP | Jieba 分词 |
| AI | DeepSeek API |

## 数据结构

- **实体数量**: 251 个
- **关系数量**: 277 条
- **关系类型**: 37 种（症状、治疗方法、病因、人群、部位等）

## 快速开始

### 1. 安装 Neo4j

```bash
# Windows
# 下载并解压 Neo4j Community Edition
# 进入 bin 目录
neo4j.bat install-service
neo4j.bat start
```

### 2. 配置环境变量

```bash
cd backend
cp .env.example .env
```

编辑 `.env` 文件，填写你的配置：
```env
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
DEEPSEEK_API_KEY=your_deepseek_api_key
```

### 3. 安装依赖

```bash
# 后端
cd backend
pip install -r requirements.txt

# 前端
cd frontend
npm install
```

### 4. 导入数据

```bash
cd backend
python import_data.py
```

### 5. 启动服务

```bash
# 启动后端 (端口 5001)
cd backend
python app.py

# 启动前端 (端口 8080)
cd frontend
npm run serve
```

访问 http://localhost:8080 开始使用。

## 使用示例

| 问题 | 回答 |
|------|------|
| 感冒的症状是什么？ | 发热, 咳嗽, 头痛, 鼻塞, 喉咙痛 |
| 高血压的治疗方法？ | 降压药, 低盐饮食 |
| 糖尿病的病因？ | 胰岛素分泌不足 |

## 项目结构

```
.
├── backend/
│   ├── app.py              # Flask 主应用
│   ├── neo4j_util.py       # Neo4j 工具类
│   ├── import_data.py      # 数据导入脚本
│   ├── requirements.txt    # Python 依赖
│   ├── .env.example        # 环境变量模板
│   └── ownthink_v2.csv     # 医疗知识数据
├── frontend/
│   ├── src/
│   │   └── App.vue         # Vue 主组件
│   └── package.json        # 前端依赖
└── README.md
```

## API 接口

### POST /query

发送问题并获取答案

```json
{
  "query": "感冒的症状是什么？"
}
```

响应：
```json
{
  "answer": "感冒的症状是发热, 咳嗽, 头痛..."
}
```

### GET /graph

获取知识图谱数据（节点和边）

## 安全说明

- ⚠️ **不要将 `.env` 文件提交到代码仓库**
- ⚠️ **请妥善保管你的 DeepSeek API Key**
- ✅ 使用 `.env.example` 作为配置模板

## 许可证

MIT License
