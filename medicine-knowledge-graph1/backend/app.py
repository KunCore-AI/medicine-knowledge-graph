from flask import Flask, request, jsonify
from flask_cors import CORS
from neo4j_util import Neo4jUtil
import jieba
from openai import OpenAI
import re
import logging
import os
from dotenv import load_dotenv

# BERT 相关模块导入
try:
    from bert_ner import extract_disease_from_query, get_bert_ner
    from bert_classifier import get_intent_classifier, classify_intent
    from sentence_encoder import get_entity_matcher, match_entity_to_graph
    BERT_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("BERT 模块导入成功")
except ImportError as e:
    BERT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"BERT 模块导入失败: {e}，将使用原有方案")

# 加载环境变量
load_dotenv()

app = Flask(__name__)
CORS(app)

# 初始化日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化 Neo4j 连接（从环境变量读取配置）
neo4j_util = Neo4jUtil(
    uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
    user=os.getenv("NEO4J_USER", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "")
)
neo4j_util._test_connection()

# 加载自定义词典（使用相对路径或环境变量）
custom_dict_path = os.getenv("CUSTOM_DICT_PATH", "custom_dict.txt")
if not os.path.isabs(custom_dict_path):
    # 如果是相对路径，使用当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    custom_dict_path = os.path.join(script_dir, custom_dict_path)

if os.path.exists(custom_dict_path):
    jieba.load_userdict(custom_dict_path)
    logger.info(f"已加载自定义词典: {custom_dict_path}")
else:
    logger.warning(f"自定义词典不存在: {custom_dict_path}")

# DeepSeek API 配置（从环境变量读取）
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    logger.error("DEEPSEEK_API_KEY 未设置，请检查 .env 文件")
    raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")

client = OpenAI(
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=deepseek_api_key
)

# 上下文管理
conversation_context = {}

# 允许的关系白名单（防止注入）- 直接使用中文关系名
ALLOWED_RELATIONS = {
    "症状", "治疗方法", "病因", "人群", "部位",  # 原有关系
    "检查项目", "推荐药物", "忌吃食物", "宜吃食物",  # 新增关系
    "并发症", "易感人群", "治疗", "预防"
}

# 同义词映射
SYNONYM_MAP = {
    "表现": "症状",
    "临床症状": "症状",
    "病症": "症状",
    "怎么治": "治疗方法",
    "如何治疗": "治疗方法",
    "怎么吃": "宜吃食物",
    "不能吃": "忌吃食物",
    "不宜吃": "忌吃食物",
    "检查": "检查项目",
    "用药": "推荐药物",
    "药物": "推荐药物",
    "药品": "推荐药物",
    "并发": "并发症",
    "后遗症": "并发症",
    "易感": "易感人群",
    "谁容易得": "易感人群",
    "怎么预防": "预防",
    "如何预防": "预防",
    "防止": "预防"
}

def map_relation_to_cypher(relation: str) -> str:
    """将中文关系同义词映射到标准关系名，并进行安全验证"""
    # 同义词映射
    mapped = SYNONYM_MAP.get(relation.strip(), relation.strip())

    # 白名单验证
    if mapped not in ALLOWED_RELATIONS:
        logger.warning(f"未知的关系类型: {relation}")
        return None
    return mapped

def call_deepseek(query, user_id="default"):
    try:
        messages = [
            {"role": "system", "content": "你是一个医学问答助手，专注于提供准确的医学知识。请根据用户的问题，提供简洁、专业的回答。如果问题涉及医学建议，请提醒用户咨询专业医生。"},
        ]
        # 初始化上下文
        if user_id not in conversation_context:
            conversation_context[user_id] = {"history": []}
        history = conversation_context[user_id].get("history", [])
        if not isinstance(history, list):
            logger.warning(f"history 不是列表，重置为 []: {history}")
            history = []
            conversation_context[user_id]["history"] = history
        messages.extend(history)
        messages.append({"role": "user", "content": query})

        logger.info(f"发送 DeepSeek API 请求: {messages}")
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        answer = completion.choices[0].message.content
        logger.info(f"DeepSeek API 返回: {answer}")

        # 更新上下文
        try:
            if user_id not in conversation_context:
                conversation_context[user_id] = {"history": []}
            new_history = conversation_context[user_id].get("history", [])
            if not isinstance(new_history, list):
                new_history = []
            new_history.append({"role": "user", "content": query})
            new_history.append({"role": "assistant", "content": answer})
            if len(new_history) > 10:
                new_history = new_history[-10:]
            conversation_context[user_id]["history"] = new_history
        except Exception as e:
            logger.error(f"更新 history 失败: {str(e)}")
            conversation_context[user_id]["history"] = []

        return answer
    except Exception as e:
        logger.error(f"调用 DeepSeek 模型失败: {str(e)}")
        return None

def extract_entity_and_relation_bert(query, user_id="default"):
    """
    使用 BERT 模型提取实体和关系（增强版）

    Args:
        query: 用户查询
        user_id: 用户ID

    Returns:
        (实体名称, 关系类型)
    """
    if not BERT_AVAILABLE:
        logger.debug("BERT 不可用，使用原有方案")
        return extract_entity_and_relation_fallback(query, user_id)

    try:
        # 使用 BERT 意图分类器
        classifier = get_intent_classifier()
        intent_result = classifier.classify(query)
        relation = intent_result["intent"]
        confidence = intent_result["confidence"]

        # 使用 BERT NER 提取实体
        entity = extract_disease_from_query(query)

        # 如果实体提取成功，尝试语义匹配到知识图谱
        if entity:
            entity_matcher = get_entity_matcher()
            # 首次使用时加载实体列表
            if entity_matcher.needs_update:
                entity_matcher.load_entities_from_neo4j(neo4j_util)
                entity_matcher.needs_update = False

            # 尝试语义匹配
            matched_entity = entity_matcher.match_entity(entity)
            if matched_entity:
                entity = matched_entity
                logger.info(f"语义匹配: '{query}' -> 实体='{entity}', 关系='{relation}' (置信度: {confidence:.2f})")
            else:
                logger.info(f"BERT 提取: '{query}' -> 实体='{entity}', 关系='{relation}' (置信度: {confidence:.2f})")

        # 如果 BERT 提取失败，回退到原有方案
        if not entity or not relation:
            logger.debug("BERT 提取不完整，使用原有方案补充")
            entity_fb, relation_fb = extract_entity_and_relation_fallback(query, user_id)
            entity = entity or entity_fb
            relation = relation or relation_fb

        return entity, relation

    except Exception as e:
        logger.error(f"BERT 提取失败: {e}，使用原有方案")
        return extract_entity_and_relation_fallback(query, user_id)


def extract_entity_and_relation_fallback(query, user_id="default"):
    # 使用 Jieba 分词
    terms = list(jieba.cut(query))
    logger.debug(f"分词结果: {terms}")

    entity = None
    relation = None
    skip_words = ["有", "哪些", "什么"]  # 跳过这些词

    # 扩展的正则表达式模式
    #分别针对"的+关系+疑问词"、"动词性询问"和"简洁连接"三种模式。
    # 通过正则表达式和后续映射，它们将自然语言问题转化为结构化查询参数
    pattern1 = r"(.+?)的(症状|治疗方法|病因|人群|部位|表现|检查项目|推荐药物|忌吃食物|宜吃食物|并发症|易感人群|治疗|预防)(是什么|有哪些|是啥|啊|呢)?"
    pattern2 = r"(.+?)(怎么办|怎么治|咋治|为啥|为什么|有啥表现|有啥症状|怎么回事|咋回事|吃什么药|吃什么|不能吃什么|宜吃什么|忌吃什么|怎么预防|如何预防|做什么检查|需要检查|有什么并发症|哪些人容易得)"
    pattern3 = r"(.+?)(症状|治疗方法|病因|人群|部位|表现|检查项目|推荐药物|忌吃食物|宜吃食物|并发症|易感人群|治疗|预防)"

    # 正则匹配
    match = None
    if re.search(pattern1, query):
        match = re.search(pattern1, query)
        entity = match.group(1).strip()
        relation = match.group(2).strip()
    elif re.search(pattern2, query):
        match = re.search(pattern2, query)
        entity = match.group(1).strip()
        ask_type = match.group(2).strip()
        ask_to_relation = {
            "怎么办": "治疗方法",
            "怎么治": "治疗方法",
            "咋治": "治疗方法",
            "为啥": "病因",
            "为什么": "病因",
            "有啥表现": "症状",
            "有啥症状": "症状",
            "怎么回事": "病因",
            "吃什么药": "推荐药物",
            "吃什么": "宜吃食物",  # 仅在不是问药时才用
            "不能吃什么": "忌吃食物",
            "宜吃什么": "宜吃食物",
            "忌吃什么": "忌吃食物",
            "怎么预防": "预防",
            "如何预防": "预防",
            "做什么检查": "检查项目",
            "需要检查": "检查项目",
            "有什么并发症": "并发症",
            "哪些人容易得": "易感人群"
        }
        relation = ask_to_relation.get(ask_type, None)
    elif re.search(pattern3, query):
        match = re.search(pattern3, query)
        entity = match.group(1).strip()
        relation = match.group(2).strip()

    # 如果正则未匹配成功，尝试现有规则
    if not (entity and relation):
        for i, term in enumerate(terms):
            if term in skip_words:
                continue
            if term == "的":
                if i > 0:
                    entity = terms[i-1]
                if i < len(terms)-1:
                    if terms[i+1] in ["症状", "治疗方法", "病因", "人群", "部位", "表现", "检查项目", "推荐药物", "忌吃食物", "宜吃食物", "并发症", "易感人群", "治疗", "预防"]:
                        relation = terms[i+1]
                        break
            elif i < len(terms)-1 and terms[i] + terms[i+1] in ["治疗方法"]:
                relation = terms[i] + terms[i+1]
                if i > 0:
                    entity = terms[i-1]
                break
            elif term in ["症状", "治疗方法", "病因", "人群", "部位", "表现", "检查项目", "推荐药物", "忌吃食物", "宜吃食物", "并发症", "易感人群", "治疗", "预防"]:
                relation = term
                if i > 0 and terms[i-1] not in skip_words:
                    entity = terms[i-1]
                break
            elif term in ["感冒", "高血压", "糖尿病", "麻疹"]:
                entity = term
                if "怎么办" in query or "怎么治" in query:
                    relation = "治疗方法"
                elif "为什么" in query:
                    relation = "病因"
                elif "表现" in query or "症状" in query:
                    relation = "症状"

    # 同义词映射
    synonym_map = {
        "表现": "症状",
        "怎么治": "治疗方法",
        "原因": "病因",
        "怎么吃": "宜吃食物",
        "不能吃": "忌吃食物",
        "检查": "检查项目",
        "药物": "推荐药物",
        "药品": "推荐药物",
        "并发": "并发症",
        "易感": "易感人群",
        "怎么预防": "预防",
        "如何预防": "预防"
    }
    if relation in synonym_map:
        relation = synonym_map[relation]

    # 上下文处理
    if not entity and user_id in conversation_context:
        entity = conversation_context[user_id].get("last_entity")

    logger.info(f"原有方案提取: entity={entity}, relation={relation}")
    return entity, relation

@app.route('/test', methods=['GET'])
def test():
    """测试路由"""
    logger.info("测试路由被调用")
    return jsonify({"status": "ok", "message": "服务器正常运行"})

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get('query', '')
    user_id = data.get('user_id', 'default')
    logger.info(f"收到的问题: {query_text}")

    # 尝试从知识图谱中提取答案（使用 BERT 增强版）
    entity, relation = extract_entity_and_relation_bert(query_text, user_id)
    if entity and relation:
        conversation_context[user_id] = {"last_entity": entity}
        try:
            # 映射关系到 Cypher 类型
            cypher_relation = map_relation_to_cypher(relation)

            # 显式执行正向和反向查询（使用参数化查询，防止注入）
            answers_forward = []
            answers_backward = []

            # 正向查询 - 使用 WHERE 子句过滤关系类型
            logger.debug(f"执行正向查询: entity={entity}, relation={cypher_relation}")
            with neo4j_util.driver.session() as session:
                query_forward_safe = """
                MATCH (n:Entity {name: $entity})-[r:RELATION]->(m:Entity)
                WHERE r.type = $relation_type
                RETURN m.name AS answer
                """
                result_forward = session.run(query_forward_safe, entity=entity.strip(), relation_type=cypher_relation)
                answers_forward = [record["answer"] for record in result_forward]
            logger.debug(f"正向查询结果: {answers_forward}")

            # 反向查询 - 使用 WHERE 子句过滤关系类型
            logger.debug(f"执行反向查询: entity={entity}, relation={cypher_relation}")
            query_backward_safe = """
            MATCH (n:Entity)-[r:RELATION]->(m:Entity {name: $entity})
            WHERE r.type = $relation_type
            RETURN n.name AS answer
            """
            with neo4j_util.driver.session() as session:
                result_backward = session.run(query_backward_safe, entity=entity.strip(), relation_type=cypher_relation)
                answers_backward = [record["answer"] for record in result_backward]
            logger.debug(f"反向查询结果: {answers_backward}")

            # 合并结果并去重
            answers = list(set(answers_forward + answers_backward))
            logger.info(f"合并查询结果: {answers}")

            if answers:
                if relation == "症状":
                    treatment = neo4j_util.get_answer(relation="治疗方法", entity=entity)
                    if treatment:
                        return jsonify({"answer": f"{entity}的{relation}是{', '.join(answers)}。建议的治疗方法是{', '.join(treatment)}"})
                return jsonify({"answer": f"{entity}的{relation}是{', '.join(answers)}"})
        except Exception as e:
            logger.error(f"知识图谱查询失败: {str(e)}")

    # 如果知识图谱无法回答，调用 DeepSeek 模型
    deepseek_answer = call_deepseek(query_text, user_id)
    if deepseek_answer:
        return jsonify({"answer": deepseek_answer})
    else:
        return jsonify({"answer": "无法回答您的问题，请尝试其他表述或咨询专业医生"})

@app.route('/graph', methods=['GET'])
def get_graph():
    try:
        with neo4j_util.driver.session() as session:
            nodes_query = """
            MATCH (n:Entity)
            RETURN n.name AS name, n.type AS type
            LIMIT 100
            """
            nodes_result = session.run(nodes_query)
            nodes = [{"id": idx, "label": record["name"], "group": record["type"]} for idx, record in enumerate(nodes_result)]

            edges_query = """
            MATCH (n:Entity)-[r:RELATION]->(m:Entity)
            WHERE n.name IN $names AND m.name IN $names
            RETURN n.name AS from_name, m.name AS to_name, r.type AS type
            """
            names = [node["label"] for node in nodes]
            edges_result = session.run(edges_query, names=names)
            edges = []
            name_to_id = {node["label"]: node["id"] for node in nodes}
            for record in edges_result:
                from_id = name_to_id.get(record["from_name"])
                to_id = name_to_id.get(record["to_name"])
                if from_id is not None and to_id is not None:
                    edges.append({"from": from_id, "to": to_id, "label": record["type"]})

        return jsonify({"nodes": nodes, "edges": edges})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    try:
        host = os.getenv("FLASK_HOST", "0.0.0.0")
        port = int(os.getenv("FLASK_PORT", "5001"))
        app.run(host=host, port=port)
    finally:
        neo4j_util.close()
