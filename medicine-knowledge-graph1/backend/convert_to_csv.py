import pandas as pd
import csv
import logging
import os
import shutil
import re

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_csv_file(file_path: str) -> pd.DataFrame:
    """读取并修复 CSV 文件，验证列名"""
    repaired_path = file_path + '.repaired.csv'
    
    # 备份文件
    backup_path = file_path + '.backup'
    if not os.path.exists(backup_path):
        shutil.copyfile(file_path, backup_path)
    
    def try_read_csv(path, encoding='utf-8'):
        try:
            return pd.read_csv(
                path, sep=',', encoding=encoding, on_bad_lines='warn',
                quoting=csv.QUOTE_MINIMAL, escapechar='\\', engine='python'
            )
        except Exception as e:
            logger.error(f"读取 {path} 失败: {str(e)}")
            return None
    
    # 尝试直接读取
    df = try_read_csv(file_path)
    if df is not None and all(col in df.columns for col in ['head', 'relation', 'tail']):
        return df.dropna(subset=['head', 'relation', 'tail'])
    
    # 修复文件
    logger.info(f"尝试修复文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as infile, \
             open(repaired_path, 'w', encoding='utf-8', newline='') as outfile:
            reader = csv.reader(infile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
            for row in reader:
                writer.writerow([str(cell).replace('\n', ' ').strip() for cell in row])
    except Exception as e:
        logger.error(f"修复失败: {str(e)}")
        return None
    
    # 读取修复后的文件
    df = try_read_csv(repaired_path, 'utf-8') or try_read_csv(repaired_path, 'gbk')
    if df is None or not all(col in df.columns for col in ['head', 'relation', 'tail']):
        logger.error("修复后仍无法读取或缺少必要列")
        return None
    return df.dropna(subset=['head', 'relation', 'tail'])

def infer_entity_type(entity: str, df: pd.DataFrame) -> str:
    """简化的实体类型推断"""
    head_rows = df[df['head'] == entity]
    tail_rows = df[df['tail'] == entity]
    
    # 疾病：head 有症状或治疗方法关系
    if any(head_rows['relation'].isin(['症状', '治疗方法'])):
        return '疾病'
    
    # 症状：tail 有症状关系或包含症状关键词
    if any(tail_rows['relation'] == '症状') or any(kw in entity for kw in ['症状', '昏迷', '烦躁']):
        return '症状'
    
    # 药物：tail 有治疗方法，包含药物关键词或正则匹配
    drug_keywords = ['药', '剂', '片', '胶囊', '注射']
    drug_patterns = [r'注射液$', r'胶囊$', r'片$', r'颗粒$']
    if any(tail_rows['relation'] == '治疗方法'):
        if any(kw in entity for kw in drug_keywords) or any(re.search(p, entity) for p in drug_patterns):
            return '药物'
    
    # 生活方式：tail 有治疗方法，包含饮食关键词
    if any(tail_rows['relation'] == '治疗方法') and any(kw in entity for kw in ['粥', '汤', '饮食']):
        return '生活方式'
    
    return '其他'

def generate_entities(df: pd.DataFrame) -> pd.DataFrame:
    """生成 entities.csv 和 jieba 词典"""
    all_entities = set(df['head']).union(set(df['tail']))
    logger.info(f"实体数：{len(all_entities)}")

    entities_data = []
    uncertain_entities = []
    for idx, entity in enumerate(all_entities):
        entity_type = infer_entity_type(entity, df)
        entities_data.append({'id': idx, 'name': entity, 'type': entity_type})
        if entity_type == '其他':
            uncertain_entities.append(entity)

    entities_df = pd.DataFrame(entities_data)
    entities_df.to_csv('entities.csv', index=False, encoding='utf-8')
    logger.info("已生成 entities.csv")

    # 生成 jieba 词典
    with open('custom_dict.txt', 'w', encoding='utf-8') as f:
        for _, row in entities_df.iterrows():
            f.write(f"{row['name']} 1000 {row['type']}\n")
    logger.info("已生成 custom_dict.txt")

    if uncertain_entities:
        with open('uncertain_entities.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(uncertain_entities))
        logger.warning(f"未分类实体：{len(uncertain_entities)} 个")

    return entities_df

def generate_relations(df: pd.DataFrame, entities_df: pd.DataFrame) -> pd.DataFrame:
    """生成 relations.csv"""
    entity_to_id = {row['name']: row['id'] for _, row in entities_df.iterrows()}
    relations_data = [
        {'source_id': entity_to_id[row['head']], 'target_id': entity_to_id[row['tail']], 'relation': row['relation']}
        for _, row in df.iterrows() if row['head'] in entity_to_id and row['tail'] in entity_to_id
    ]
    relations_df = pd.DataFrame(relations_data)
    relations_df.to_csv('relations.csv', index=False, encoding='utf-8')
    logger.info("已生成 relations.csv")
    return relations_df

def main():
    """主函数"""
    file_path = r'D:\Desktop\文件\课程文件\人工知识图谱\医疗知识问答系统\medicine-knowledge-graph1\backend\ownthink_v2.csv'
    df = read_csv_file(file_path)
    if df is None:
        return
    
    entities_df = generate_entities(df)
    generate_relations(df, entities_df)

if __name__ == '__main__':
    main()