import pandas as pd
import logging
from neo4j import GraphDatabase
from pathlib import Path
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_logging(log_file='neo4j_import.log'):
    """设置日志文件，将日志同时输出到控制台和指定文件。
    
    Args:
        log_file (str): 日志文件的路径，默认为 'neo4j_import.log'。
    """
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

class Neo4jImporter:
    """Neo4j 数据导入工具类，用于从 CSV 文件导入实体和关系数据到 Neo4j 数据库。"""
    
    def __init__(self, uri="neo4j://localhost:7687", user="neo4j", password="service-vatican-status-alfonso-regard-1763"):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info("成功连接到 Neo4j 数据库")
        except Exception as e:
            logger.error(f"无法连接到 Neo4j 数据库：{str(e)}")
            raise
        setup_logging()

    def close(self):
        """关闭 Neo4j 数据库连接。
        
        Raises:
            Exception: 如果关闭连接失败，记录错误日志。
        """
        try:
            self.driver.close()
            logger.info("Neo4j 连接已关闭")
        except Exception as e:
            logger.error(f"关闭 Neo4j 连接失败：{str(e)}")

    def clear_database(self):
        """清空 Neo4j 数据库，删除所有节点和关系。
        
        Raises:
            Exception: 如果清空数据库失败，抛出异常并记录错误日志。
        """
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("数据库已清空")
        except Exception as e:
            logger.error(f"清空数据库失败：{str(e)}")
            raise

    def infer_entity_type(self, entity, df, head_col='head', tail_col='tail', relation_col='relation'):
        """
        根据实体在 CSV 数据中的关系动态推断其类型（如疾病、症状、药物等）。
        Returns:
            str: 推断的实体类型（如 '疾病', '症状', '药物'），如果无法推断，返回 '未知'。
        Raises:
            Exception: 如果推断过程失败，记录警告日志并返回 '未知'。
        """
        try:
            head_rows = df[df[head_col] == entity]
            tail_rows = df[df[tail_col] == entity]
            
            # 疾病：作为症状或病因的头实体
            if any(head_rows[relation_col].isin(['症状', '病因'])):
                return '疾病'
            # 症状：作为症状关系的尾实体
            if any(tail_rows[relation_col] == '症状'):
                return '症状'
            # 药物：包含药物关键词或常见药物名称，且作为治疗方法的尾实体
            drug_keywords = ['药', '剂', '素', '油']
            common_drugs = ['硝酸甘油', '阿司匹林', '质子泵抑制剂', '支气管扩张剂', '抗生素', '抗病毒药物']
            if any(tail_rows[relation_col] == '治疗方法'):
                if any(kw in entity for kw in drug_keywords) or entity in common_drugs:
                    return '药物'
            # 生活方式：包含生活方式关键词，且作为治疗方法的尾实体
            lifestyle_keywords = ['饮食', '喝水', '休息']
            if any(tail_rows[relation_col] == '治疗方法') and any (kw in entity for kw in lifestyle_keywords):
                return '生活方式'
            # 医疗程序：包含医疗程序关键词，且作为治疗方法的尾实体
            procedure_keywords = ['手术', '治疗', '支架', '固定', '冲洗', '注射', '疗']
            if any(tail_rows[relation_col] == '治疗方法') and any(kw in entity for kw in procedure_keywords):
                return '医疗程序'
            # 其他干预：作为治疗方法的尾实体，但不属于药物或医疗程序
            if any(tail_rows[relation_col] == '治疗方法'):
                return '其他干预'
            # 病因：作为病因关系的尾实体
            if any(tail_rows[relation_col] == '病因'):
                return '病因'
            # 部位：作为部位关系的尾实体
            if any(tail_rows[relation_col] == '部位'):
                return '部位'
            # 人群：作为人群关系的尾实体
            if any(tail_rows[relation_col] == '人群'):
                return '人群'
            return '未知'
        except Exception as e:
            logger.warning(f"推断实体类型 {entity} 失败：{str(e)}")
            return '未知'

    def read_csv_file(self, csv_file, separator=',', expected_columns=['head', 'relation', 'tail']):
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                header = f.readline().strip().split(separator)
            logger.info(f"读取的列名：{header}")
            
            if header != expected_columns:
                logger.error(f"CSV 文件列名不匹配，期望 {expected_columns}，实际为 {header}")
                raise ValueError(f"列名不匹配：{header}")
            
            chunks = []
            chunk_size = 100000
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size, encoding='utf-8', sep=separator, 
                                   skiprows=1, names=expected_columns, on_bad_lines='skip'):
                logger.info(f"处理数据块：{len(chunk)} 条三元组")
                chunk = chunk.dropna(subset=expected_columns)
                chunks.append(chunk)
            
            if not chunks:
                logger.error("CSV 文件为空或无有效数据")
                raise ValueError("无有效数据")
            
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"总计读取 {len(df)} 条三元组")
            return df
        except FileNotFoundError:
            logger.error(f"文件 {csv_file} 不存在")
            raise
        except UnicodeDecodeError:
            logger.error(f"文件 {csv_file} 编码错误，尝试其他编码（如 'gbk'）")
            raise
        except Exception as e:
            logger.error(f"读取文件 {csv_file} 失败：{str(e)}")
            raise

    def import_data(self, csv_file):
        """
        从 CSV 文件导入实体和关系数据到 Neo4j 数据库。
        
        Args:
            csv_file (str): 包含三元组数据的 CSV 文件路径。
        
        Raises:
            Exception: 如果导入过程失败，抛出异常并记录错误日志。
        """
        try:
            # 读取 CSV 文件
            df = self.read_csv_file(csv_file)
            logger.info(f"总计关系类型：{len(df['relation'].unique())}")
            logger.info(f"导入的关系类型：{df['relation'].unique()}")
            if df.empty:
                logger.warning("无有效三元组，跳过导入")
                return
            
            logger.info("清理后数据（前 5 行）：")
            logger.info(f"\n{df.head().to_string()}")

            # 提取所有唯一实体
            entities = set(df['head']).union(set(df['tail']))
            logger.info(f"提取的实体数：{len(entities)}")
            
            # 推断实体类型
            entity_types = {}
            uncertain_entities = []
            for entity in entities:
                if pd.isna(entity):
                    continue
                entity_type = self.infer_entity_type(entity, df)
                entity_types[entity] = entity_type
                if entity_type == '未知':
                    uncertain_entities.append(entity)

            # 记录未明确分类的实体
            if uncertain_entities:
                with open('uncertain_entities.txt', 'w', encoding='utf-8') as f:
                    f.write('\n'.join(uncertain_entities))
                logger.warning(f"发现 {len(uncertain_entities)} 个未明确分类的实体，记录在 uncertain_entities.txt")

            # 导入实体
            with self.driver.session() as session:
                for entity, entity_type in entity_types.items():
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type
                        """,
                        name=str(entity), type=entity_type
                    )
            logger.info("实体导入完成")

            # 导入关系
            with self.driver.session() as session:
                for _, row in df.iterrows():
                    head = row['head']
                    relation = row['relation']
                    tail = row['tail']
                    if pd.isna(head) or pd.isna(tail):
                        continue
                    cypher = f"""
                        MATCH (source:Entity {{name: $source_name}})
                        MATCH (target:Entity {{name: $target_name}})
                        MERGE (source)-[r:`{relation}`]->(target)
                    """
                    session.run(cypher, source_name=str(head), target_name=str(tail))
            logger.info("关系导入完成")

            # 验证导入结果
            with self.driver.session() as session:
                result = session.run("MATCH (n:Entity) RETURN count(n) AS entity_count")
                entity_count = result.single()['entity_count']
                result = session.run("MATCH ()-[r]->() RETURN count(r) AS relation_count")
                relation_count = result.single()['relation_count']
                logger.info(f"导入验证：{entity_count} 个实体，{relation_count} 条关系")
                if entity_count != len(entity_types):
                    logger.warning(f"实体数量不匹配，预期 {len(entity_types)}，实际 {entity_count}")
                if relation_count != len(df):
                    logger.warning(f"关系数量不匹配，预期 {len(df)}，实际 {relation_count}")

        except Exception as e:
            logger.error(f"数据导入失败：{str(e)}")
            raise

if __name__ == "__main__":
    """主程序入口，执行数据导入流程。"""
    importer = Neo4jImporter()
    try:
        importer.clear_database()
        importer.import_data(r"D:\Desktop\文件\课程文件\人工知识图谱\医疗知识问答系统\medicine-knowledge-graph1\backend\ownthink_v2.csv")
    except Exception as e:
        logger.error(f"导入失败：{str(e)}")
    finally:
        importer.close()