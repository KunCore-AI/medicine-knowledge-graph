#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT 意图分类模块
使用 BERT 模型分类用户查询意图（症状/治疗/预防/饮食/检查/并发症等）
"""

import os
import logging
import re
from typing import List, Dict, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 意图标签定义
INTENT_LABELS = {
    0: "症状",        # 症状、表现、征兆
    1: "治疗方法",    # 怎么治、治疗、怎么处理
    2: "病因",        # 原因、为什么、病因
    3: "预防",        # 预防、怎么防止、如何避免
    4: "宜吃食物",    # 能吃什么、建议吃
    5: "忌吃食物",    # 不能吃什么、禁忌
    6: "推荐药物",    # 吃什么药、用药
    7: "检查项目",    # 做什么检查、检查
    8: "并发症",      # 并发症、后遗症、会引发什么
    9: "易感人群",    # 谁容易得、哪些人
}

# 意图同义词映射（用于规则分类）
# 注意：更具体的匹配项放在前面优先检查
INTENT_SYNONYMS = {
    "症状": ["症状", "表现", "征兆", "现象", "感觉", "不适", "症状有哪些", "有什么症状", "症状是"],
    "治疗方法": ["怎么治", "如何治疗", "治疗方法", "怎么办", "处理", "治疗", "医治", "诊治"],
    "病因": ["为什么", "病因", "原因", "怎么回事", "怎么会", "引起", "导致", "由于"],
    "预防": ["怎么预防", "如何预防", "预防", "防止", "避免", "预防措施"],
    "推荐药物": ["吃什么药", "用药", "药物", "药品", "服用", "吃啥药", "药物推荐"],  # 移除单独的"药物"避免误匹配
    "宜吃食物": ["能吃什么", "宜吃", "建议吃", "适合吃", "可以吃", "饮食", "可以食用", "适宜吃"],  # 移除"吃什么"
    "忌吃食物": ["不能吃", "忌吃", "禁忌", "不适合吃", "不可以吃", "不能吃什么", "不可以食用"],
    "检查项目": ["做什么检查", "检查", "化验", "检测", "诊断"],
    "并发症": ["并发症", "后遗症", "会引发", "会导致", "并发"],
    "易感人群": ["谁容易得", "哪些人", "易感", "好发于", "多发于", "人群"],
}


class BERTIntentClassifier:
    """BERT 意图分类器"""

    def __init__(self, model_name: str = "hfl/chinese-bert-wwm-ext"):
        """
        初始化 BERT 意图分类器

        Args:
            model_name: 预训练模型名称
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.is_loaded = False

        # 可以选择加载微调后的模型，这里先使用规则方法
        logger.info(f"意图分类器初始化，设备: {self.device}")

    def load_model(self, model_path: str = None):
        """
        加载微调后的模型

        Args:
            model_path: 模型路径，如果为 None 则使用预训练模型
        """
        try:
            model_path = model_path or self.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(INTENT_LABELS)
            )
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"BERT 意图分类模型加载成功: {model_path}")
        except Exception as e:
            logger.warning(f"BERT 意图分类模型加载失败: {e}，将使用规则分类")
            self.is_loaded = False

    def predict(self, text: str) -> Tuple[str, float]:
        """
        预测查询意图

        Args:
            text: 查询文本

        Returns:
            (意图标签, 置信度)
        """
        if not self.is_loaded:
            return self._rule_based_classify(text)

        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                confidence, pred_id = torch.max(probs, dim=-1)

            intent = INTENT_LABELS[pred_id.item()]
            confidence = confidence.item()

            logger.debug(f"BERT 分类: {text} -> {intent} (置信度: {confidence:.2f})")
            return intent, confidence

        except Exception as e:
            logger.error(f"BERT 分类失败: {e}，使用规则分类")
            return self._rule_based_classify(text)

    def _rule_based_classify(self, text: str) -> Tuple[str, float]:
        """
        基于规则的意图分类（备用方案）

        Args:
            text: 查询文本

        Returns:
            (意图标签, 置信度)
        """
        text_lower = text.lower()

        # 检查每个意图的关键词
        for intent, keywords in INTENT_SYNONYMS.items():
            for keyword in keywords:
                if keyword in text:
                    # 计算简单的置信度（基于关键词匹配程度）
                    confidence = 0.8 if f"{intent}是什么" in text or f"{intent}有哪些" in text else 0.7
                    logger.debug(f"规则分类: {text} -> {intent} (关键词: {keyword})")
                    return intent, confidence

        # 默认返回症状（最常见的查询类型）
        logger.debug(f"规则分类: {text} -> 症状 (默认)")
        return "症状", 0.5


class EnhancedIntentClassifier:
    """增强版意图分类器（结合 BERT 和规则）"""

    def __init__(self):
        self.bert_classifier = BERTIntentClassifier()

        # 可选：加载微调的模型
        # self.bert_classifier.load_model("path/to/fine-tuned-model")

    def classify(self, text: str) -> Dict:
        """
        分类查询意图

        Args:
            text: 查询文本

        Returns:
            {
                "intent": "症状",
                "confidence": 0.85,
                "raw_query": text
            }
        """
        intent, confidence = self.bert_classifier.predict(text)

        return {
            "intent": intent,
            "confidence": confidence,
            "raw_query": text
        }

    def extract_entity_and_intent(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        从查询中同时提取实体和意图（高层接口）

        Args:
            query: 用户查询

        Returns:
            (实体名称, 意图标签)
        """
        # 获取意图
        intent_result = self.classify(query)
        intent = intent_result["intent"]

        # 提取实体（这里可以结合 bert_ner）
        from bert_ner import extract_disease_from_query
        entity = extract_disease_from_query(query)

        return entity, intent


# 全局单例
_intent_classifier_instance = None


def get_intent_classifier() -> EnhancedIntentClassifier:
    """获取意图分类器实例（单例）"""
    global _intent_classifier_instance
    if _intent_classifier_instance is None:
        _intent_classifier_instance = EnhancedIntentClassifier()
    return _intent_classifier_instance


def classify_intent(query: str) -> str:
    """
    分类查询意图（简化接口）

    Args:
        query: 用户查询

    Returns:
        意图标签
    """
    classifier = get_intent_classifier()
    result = classifier.classify(query)
    return result["intent"]


if __name__ == "__main__":
    # 测试代码
    print("=== 意图分类器测试 ===")

    classifier = EnhancedIntentClassifier()

    test_queries = [
        "糖尿病的症状是什么？",
        "高血压怎么治疗？",
        "感冒能吃什么？",
        "哮喘不能吃什么？",
        "心脏病需要做什么检查？",
        "中风的并发症有哪些？",
        "哪些人容易得糖尿病？",
        "感冒怎么预防？",
        "肺炎是什么原因引起的？",
    ]

    for query in test_queries:
        result = classifier.classify(query)
        print(f"查询: {query}")
        print(f"意图: {result['intent']} (置信度: {result['confidence']:.2f})")
        print()
