from neo4j import GraphDatabase
import logging
from typing import List, Optional
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class Neo4jUtil:
    """Neo4j 数据库操作工具类，用于连接 Neo4j 数据库并执行查询操作。"""

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        # 优先使用传入的参数，否则从环境变量读取
        self.uri = uri or os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")

        if not self.password:
            logger.warning("Neo4j 密码未设置，请检查 NEO4J_PASSWORD 环境变量")

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._test_connection()
            logger.info("Neo4j 连接初始化成功")
        except Exception as e:
            logger.error(f"Neo4j 连接初始化失败: {str(e)}")
            raise

    def _test_connection(self) -> None:
        """测试 Neo4j 连接并验证数据库状态。"""
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS count")
            count = result.single()["count"]
            logger.info(f"Neo4j 连接成功，数据库中有 {count} 个节点")

    def close(self) -> None:
        """关闭 Neo4j 数据库连接，释放资源。"""
        try:
            if self.driver:
                self.driver.close()
                logger.info("Neo4j 连接已关闭")
        except Exception as e:
            logger.error(f"关闭 Neo4j 连接失败: {str(e)}")

    def get_answer(self, relation: str, entity: str) -> List[str]:
        """根据关系和实体查询答案（支持双向查询）"""

        answers = []
        try:
            # 映射关系到 Cypher 类型（通过白名单验证）
            mapped_relation = map_relation_to_cypher(relation)
            if not mapped_relation:
                logger.warning(f"无效的关系类型: {relation}")
                return []

            with self.driver.session() as session:
                # 正向查询：entity -> relation -> target
                # 使用 WHERE 子句过滤关系类型（实际关系存储在 r.type 属性中）
                query_forward = """
                MATCH (n:Entity {name: $entity})-[r:RELATION]->(m:Entity)
                WHERE r.type = $relation_type
                RETURN m.name AS answer
                """
                logger.debug(f"正向查询参数: entity={entity}, relation={mapped_relation}")
                result_forward = session.run(query_forward, entity=entity.strip(), relation_type=mapped_relation)
                answers_forward = [record["answer"] for record in result_forward]
                logger.debug(f"正向查询结果: {answers_forward}")

                # 反向查询：target -> relation -> entity
                query_backward = """
                MATCH (n:Entity)-[r:RELATION]->(m:Entity {name: $entity})
                WHERE r.type = $relation_type
                RETURN n.name AS answer
                """
                logger.debug(f"反向查询参数: entity={entity}, relation={mapped_relation}")
                result_backward = session.run(query_backward, entity=entity.strip(), relation_type=mapped_relation)
                answers_backward = [record["answer"] for record in result_backward]
                logger.debug(f"反向查询结果: {answers_backward}")

                # 合并结果并去重
                answers = list(set(answers_forward + answers_backward))
                logger.info(f"合并查询结果: {answers}")

            return answers
        except Exception as e:
            logger.error(f"Neo4j 查询失败: {str(e)}")
            return []

    def __del__(self):
        """析构函数，确保连接关闭。"""
        self.close()
