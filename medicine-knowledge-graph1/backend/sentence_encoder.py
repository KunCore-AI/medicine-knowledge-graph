#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义匹配模块
使用 Sentence-BERT (SBERT) 计算文本相似度，实现模糊匹配和同义词识别
"""

import os
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SemanticMatcher:
    """语义匹配器，使用 SBERT 计算相似度"""

    def __init__(self, model_name: str = None):
        """
        初始化语义匹配器

        Args:
            model_name: Sentence Transformer 模型名称或路径
        """
        # 优先使用本地模型
        local_model = r"D:\Desktop\文件\课程文件\人工知识图谱\医疗知识问答系统\medicine-knowledge-graph1\backend\models\paraphrase-multilingual-MiniLM-L12-v2"

        self.model_name = model_name or local_model
        self.model = None
        self.entity_embeddings = {}  # 缓存实体嵌入
        self.is_loaded = False

        try:
            logger.info(f"正在加载 Sentence Transformer 模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.is_loaded = True
            logger.info("Sentence Transformer 模型加载成功")
        except Exception as e:
            logger.warning(f"Sentence Transformer 模型加载失败: {e}，将使用精确匹配")
            self.is_loaded = False

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本编码为向量

        Args:
            texts: 文本列表

        Returns:
            向量数组 (n_texts, embedding_dim)
        """
        if not self.is_loaded or not texts:
            return None

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            return None

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度分数 (0-1)
        """
        if not self.is_loaded:
            # 降级为字符串相似度
            return self._string_similarity(text1, text2)

        try:
            embeddings = self.encode([text1, text2])
            if embeddings is None:
                return self._string_similarity(text1, text2)

            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return self._string_similarity(text1, text2)

    def find_best_match(
        self,
        query: str,
        candidates: List[str],
        threshold: float = 0.6
    ) -> Tuple[Optional[str], float]:
        """
        从候选列表中找到最佳匹配

        Args:
            query: 查询文本
            candidates: 候选文本列表
            threshold: 相似度阈值

        Returns:
            (最佳匹配, 相似度分数)
        """
        if not candidates:
            return None, 0.0

        # 首先尝试精确匹配
        if query in candidates:
            return query, 1.0

        # 使用语义匹配
        if self.is_loaded:
            try:
                query_embedding = self.encode([query])
                candidate_embeddings = self.encode(candidates)

                if query_embedding is not None and candidate_embeddings is not None:
                    similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
                    best_idx = np.argmax(similarities)
                    best_score = similarities[best_idx]

                    if best_score >= threshold:
                        return candidates[best_idx], float(best_score)
            except Exception as e:
                logger.error(f"语义匹配失败: {e}")

        # 降级为字符串相似度
        best_match = None
        best_score = 0.0
        for candidate in candidates:
            score = self._string_similarity(query, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate

        if best_score >= threshold * 0.8:  # 字符串匹配使用稍低的阈值
            return best_match, best_score

        return None, best_score

    def _string_similarity(self, text1: str, text2: str) -> float:
        """
        字符串相似度（备用方案）

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度分数 (0-1)
        """
        # 简单的包含关系检查
        if text1 in text2 or text2 in text1:
            return 0.9

        # 编辑距离相似度
        len1, len2 = len(text1), len(text2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # 简化的相似度计算
        common = set(text1) & set(text2)
        total = set(text1) | set(text2)
        return len(common) / len(total) if total else 0.0


class EntityMatcher:
    """实体匹配器，用于匹配用户查询与知识图谱中的实体"""

    def __init__(self):
        self.semantic_matcher = SemanticMatcher()
        self.known_entities = set()  # 从知识图谱加载的实体
        self.entity_embeddings = None
        self.needs_update = True

    def load_entities_from_neo4j(self, neo4j_util):
        """
        从 Neo4j 加载所有实体

        Args:
            neo4j_util: Neo4jUtil 实例
        """
        try:
            with neo4j_util.driver.session() as session:
                result = session.run("MATCH (n:Entity) RETURN n.name AS name")
                entities = [record["name"] for record in result]

            self.known_entities = set(entities)
            self.needs_update = True

            logger.info(f"从 Neo4j 加载了 {len(entities)} 个实体")

        except Exception as e:
            logger.error(f"加载实体失败: {e}")

    def match_entity(self, query_entity: str, threshold: float = 0.7) -> Optional[str]:
        """
        匹配查询实体到知识图谱中的实体

        Args:
            query_entity: 用户查询中的实体
            threshold: 相似度阈值

        Returns:
            匹配到的实体名称，如果未找到返回 None
        """
        # 精确匹配
        if query_entity in self.known_entities:
            return query_entity

        # 语义匹配
        candidates = list(self.known_entities)
        best_match, score = self.semantic_matcher.find_best_match(
            query_entity,
            candidates,
            threshold
        )

        if best_match:
            logger.info(f"实体匹配: '{query_entity}' -> '{best_match}' (相似度: {score:.2f})")

        return best_match

    def get_similar_entities(
        self,
        entity: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        获取与给定实体相似的实体列表

        Args:
            entity: 实体名称
            top_k: 返回前 K 个相似实体

        Returns:
            [(实体名, 相似度), ...]
        """
        if not self.semantic_matcher.is_loaded:
            return []

        candidates = list(self.known_entities)
        if entity in candidates:
            candidates.remove(entity)

        if not candidates:
            return []

        try:
            query_embedding = self.semantic_matcher.encode([entity])
            candidate_embeddings = self.semantic_matcher.encode(candidates)

            if query_embedding is None or candidate_embeddings is None:
                return []

            similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

            # 获取 top_k
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = [
                (candidates[i], float(similarities[i]))
                for i in top_indices
                if similarities[i] > 0.5  # 只返回有一定相似度的
            ]

            return results

        except Exception as e:
            logger.error(f"获取相似实体失败: {e}")
            return []


# 全局单例
_entity_matcher_instance = None


def get_entity_matcher() -> EntityMatcher:
    """获取实体匹配器实例（单例）"""
    global _entity_matcher_instance
    if _entity_matcher_instance is None:
        _entity_matcher_instance = EntityMatcher()
    return _entity_matcher_instance


def match_entity_to_graph(query_entity: str, neo4j_util=None) -> Optional[str]:
    """
    将查询实体匹配到知识图谱中的实体（高层接口）

    Args:
        query_entity: 用户查询中的实体
        neo4j_util: Neo4jUtil 实例（用于加载实体列表）

    Returns:
        匹配到的实体名称
    """
    matcher = get_entity_matcher()

    # 首次使用时加载实体
    if matcher.needs_update and neo4j_util:
        matcher.load_entities_from_neo4j(neo4j_util)
        matcher.needs_update = False

    return matcher.match_entity(query_entity)


if __name__ == "__main__":
    # 测试代码
    print("=== 语义匹配测试 ===")

    matcher = SemanticMatcher()

    # 测试相似度计算
    pairs = [
        ("糖尿病", "糖代谢病"),
        ("高血压", "血压高"),
        ("感冒", "上呼吸道感染"),
        ("心脏病", "心脏疾病"),
    ]

    for text1, text2 in pairs:
        similarity = matcher.compute_similarity(text1, text2)
        print(f"'{text1}' vs '{text2}': {similarity:.2f}")

    print("\n=== 实体匹配测试 ===")

    # 模拟知识图谱实体
    entity_matcher = EntityMatcher()
    entity_matcher.known_entities = {
        "糖尿病", "高血压", "冠心病", "感冒", "哮喘",
        "肺炎", "胃炎", "肝炎", "肾炎", "贫血"
    }

    test_queries = ["血压高", "糖代谢病", "上呼吸道感染", "心梗"]
    for query in test_queries:
        match = entity_matcher.match_entity(query)
        print(f"查询: '{query}' -> 匹配: '{match}'")
