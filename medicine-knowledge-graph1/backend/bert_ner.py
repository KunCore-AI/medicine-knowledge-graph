#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT NER 实体识别模块
使用预训练的中文 BERT 模型识别医疗实体（疾病名称）
"""

import os
import logging
from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 医疗领域常用的 NER 模型（中文）
# 使用 hfl/chinese-ner 相关模型或通用中文 BERT
MODEL_OPTIONS = [
    "hfl/chinese-bert-wwm-ext",  # 中文 BERT 全词掩码
    "bert-base-chinese",          # 通用中文 BERT
]


class BERTNER:
    """BERT 命名实体识别类，用于从查询中提取疾病等医疗实体"""

    def __init__(self, model_name: str = None):
        """
        初始化 BERT NER 模型

        Args:
            model_name: 模型名称，默认使用本地下载的模型
        """
        # 优先使用本地NER模型（真正的NER模型，专门训练过）
        local_ner_model = r"D:\Desktop\文件\课程文件\人工知识图谱\医疗知识问答系统\medicine-knowledge-graph1\backend\models\roberta-base-finetuned-cluener2020-chinese"

        self.model_name = model_name or local_ner_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        try:
            # 使用 Hugging Face 的 NER pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)

            # 创建 NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",  # 合并同一实体的子词
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"BERT NER 模型加载成功: {self.model_name}")

        except Exception as e:
            logger.warning(f"BERT NER 模型加载失败: {e}，将使用备用方案")
            self.ner_pipeline = None

    def extract_entities(self, text: str, entity_types: List[str] = None) -> List[str]:
        """
        从文本中提取实体

        Args:
            text: 输入文本
            entity_types: 要提取的实体类型列表（如 ['ORG', 'LOC']）

        Returns:
            提取的实体列表
        """
        if not text or not text.strip():
            return []

        # 如果模型加载失败，返回空列表（由备用方案处理）
        if self.ner_pipeline is None:
            return []

        try:
            results = self.ner_pipeline(text)

            # 过滤指定类型的实体
            entities = []
            for result in results:
                if entity_types is None or result.get('entity_group') in entity_types:
                    entities.append(result['word'])

            logger.debug(f"BERT NER 提取结果: {entities}")
            return entities

        except Exception as e:
            logger.error(f"BERT NER 推理失败: {e}")
            return []

    def extract_disease_entities(self, text: str) -> List[str]:
        """
        专门提取疾病实体
        由于通用 NER 模型可能没有专门的疾病标签，
        这里结合关键词匹配和 NER 结果

        Args:
            text: 输入文本

        Returns:
            疾病实体列表
        """
        # 首先尝试使用 NER
        ner_entities = self.extract_entities(text)

        # 常见疾病关键词（用于增强，从知识图谱中提取）
        disease_keywords = [
            # 呼吸系统
            '感冒', '流感', '肺炎', '哮喘', '鼻炎', '咽炎', '支气管炎', '肺结核',
            # 消化系统
            '胃炎', '肠炎', '肝炎', '肾炎', '胃溃疡', '十二指肠溃疡',
            # 循环系统
            '高血压', '低血压', '心脏病', '冠心病', '心梗', '心肌梗死', '中风', '脑梗',
            # 内分泌
            '糖尿病', '甲状腺功能亢进', '甲亢', '甲状腺功能减退',
            # 血液系统
            '贫血', '白血病', '血小板减少症',
            # 神经系统
            '偏头痛', '癫痫', '帕金森', '阿尔茨海默症',
            # 精神心理
            '抑郁症', '焦虑症', '精神分裂症', '失眠症',
            # 皮肤病
            '湿疹', '皮炎', '荨麻疹', '痤疮', '过敏',
            # 风湿免疫
            '关节炎', '风湿', '类风湿关节炎', '痛风', '强直性脊柱炎',
            # 骨科
            '骨质疏松', '骨折', '腰椎间盘突出', '颈椎病',
            # 传染病
            '麻疹', '水痘', '风疹', '腮腺炎', '伤寒', '痢疾', '乙肝', '肺结核',
            # 五官科
            '结膜炎', '角膜炎', '中耳炎', '扁桃体炎',
            # 妇科
            '宫颈炎', '盆腔炎', '乳腺增生', '子宫肌瘤',
            # 儿科
            '手足口病', '百日咳',
            # 泌尿系统
            '尿路感染', '肾结石', '膀胱炎',
            # 肿瘤
            '癌症', '肿瘤', '肺癌', '胃癌', '肝癌', '肠癌', '乳腺癌',
            # 症状类（可能被查询但不是疾病）
            # '发烧', '发热', '咳嗽', '头痛' - 这些会通过NER的'address'等类型提取
        ]

        # 合并 NER 结果和关键词匹配
        entities = list(set(ner_entities))
        for keyword in disease_keywords:
            if keyword in text and keyword not in entities:
                entities.append(keyword)

        # 按文本中出现顺序排序
        entities.sort(key=lambda x: text.find(x) if x in text else len(text))

        return entities


# 备用方案：基于规则的实体提取（当 BERT 不可用时）
class RuleBasedNER:
    """基于规则的实体提取（备用方案）"""

    def __init__(self):
        # 从数据库加载的实体列表（可以动态更新）
        self.entity_list = set()

    def update_entity_list(self, entities: List[str]):
        """更新实体列表"""
        self.entity_list = set(entities)

    def extract_entities(self, text: str) -> List[str]:
        """从文本中提取实体"""
        found = []
        for entity in self.entity_list:
            if entity in text:
                found.append(entity)
        # 按长度排序，优先匹配长的实体名
        found.sort(key=lambda x: len(x), reverse=True)
        return found


# 全局单例
_bert_ner_instance = None
_rule_based_ner = RuleBasedNER()


def get_bert_ner() -> BERTNER:
    """获取 BERT NER 实例（单例）"""
    global _bert_ner_instance
    if _bert_ner_instance is None:
        _bert_ner_instance = BERTNER()
    return _bert_ner_instance


def get_rule_based_ner() -> RuleBasedNER:
    """获取基于规则的 NER 实例"""
    return _rule_based_ner


def extract_disease_from_query(query: str) -> Optional[str]:
    """
    从查询中提取疾病实体（高层接口）
    优先使用 BERT，失败时使用规则方法

    Args:
        query: 用户查询

    Returns:
        提取的疾病名称，如果未找到返回 None
    """
    # 首先尝试 BERT
    bert_ner = get_bert_ner()
    diseases = bert_ner.extract_disease_entities(query)

    if diseases:
        return diseases[0]  # 返回第一个（最可能是主语）

    # 备用：基于规则的方法
    rule_based = get_rule_based_ner()
    diseases = rule_based.extract_entities(query)

    return diseases[0] if diseases else None


if __name__ == "__main__":
    # 测试代码
    print("=== BERT NER 测试 ===")

    ner = BERTNER()

    test_queries = [
        "糖尿病的症状是什么？",
        "高血压怎么治疗？",
        "感冒需要吃什么药？",
        "我有哮喘，应该注意什么？",
    ]

    for query in test_queries:
        diseases = ner.extract_disease_entities(query)
        print(f"查询: {query}")
        print(f"提取的疾病: {diseases}")
        print()
