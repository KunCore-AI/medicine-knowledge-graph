#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT 模型下载脚本
手动下载 Hugging Face 模型到本地目录
"""

import os
import sys
from pathlib import Path

# 设置使用国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 模型下载目标目录
MODELS_DIR = Path(r"D:\Desktop\文件\课程文件\人工知识图谱\医疗知识问答系统\medicine-knowledge-graph1\backend\models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 需要下载的模型列表
MODELS_TO_DOWNLOAD = [
    {
        "name": "hfl/chinese-bert-wwm-ext",
        "local_dir": MODELS_DIR / "chinese-bert-wwm-ext",
        "description": "中文 BERT NER 模型"
    },
    {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "local_dir": MODELS_DIR / "paraphrase-multilingual-MiniLM-L12-v2",
        "description": "多语言语义匹配模型"
    }
]


def download_model(model_name, local_dir):
    """下载单个模型"""
    print(f"\n{'='*60}")
    print(f"下载模型: {model_name}")
    print(f"目标目录: {local_dir}")
    print(f"{'='*60}")

    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
        from sentence_transformers import SentenceTransformer

        # 下载 BERT 模型
        if "chinese-bert" in model_name:
            print("下载 Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(local_dir)

            print("下载模型文件...")
            model = AutoModel.from_pretrained(model_name)
            model.save_pretrained(local_dir)

        # 下载 Sentence Transformer 模型
        elif "sentence-transformers" in model_name:
            print("下载 Sentence Transformer...")
            model = SentenceTransformer(model_name)
            # 转换为字符串路径
            model.save(str(local_dir))

        print(f"[OK] 模型下载成功: {local_dir}")
        return True

    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        return False


def download_with_huggingface_cli():
    """使用 huggingface-cli 下载（备用方法）"""
    print("\n尝试使用 huggingface-cli 下载...")

    for model_info in MODELS_TO_DOWNLOAD:
        model_name = model_info["name"]
        local_dir = model_info["local_dir"]

        cmd = f"huggingface-cli download {model_name} --local-dir {local_dir} --local-dir-use-symlinks False"
        print(f"\n执行: {cmd}")

        try:
            import subprocess
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[OK] {model_info['description']} 下载成功")
            else:
                print(f"[ERROR] 下载失败: {result.stderr}")
        except Exception as e:
            print(f"[ERROR] 执行失败: {e}")


def main():
    """主函数"""
    print("="*60)
    print("BERT 模型下载工具")
    print("="*60)
    print(f"\n模型将保存到: {MODELS_DIR}")

    # 检查是否安装了必要的库
    try:
        import transformers
        import sentence_transformers
        print(f"\n[OK] transformers 版本: {transformers.__version__}")
        print(f"[OK] sentence-transformers 版本: {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"\n[ERROR] 缺少必要的库: {e}")
        print("请先运行: pip install transformers sentence-transformers torch")
        return

    # 选择下载方式
    print("\n请选择下载方式:")
    print("1. Python API 下载")
    print("2. huggingface-cli 下载")
    print("3. 两者都尝试")

    choice = input("\n请输入选项 (1/2/3) [默认: 1]: ").strip() or "1"

    success_count = 0

    if choice in ["1", "3"]:
        print("\n方式 1: 使用 Python API 下载")
        for model_info in MODELS_TO_DOWNLOAD:
            if download_model(model_info["name"], model_info["local_dir"]):
                success_count += 1

    if choice in ["2", "3"] and success_count < len(MODELS_TO_DOWNLOAD):
        print("\n方式 2: 使用 huggingface-cli 下载")
        download_with_huggingface_cli()

    # 显示结果
    print("\n" + "="*60)
    print("下载完成!")
    print("="*60)

    for model_info in MODELS_TO_DOWNLOAD:
        local_dir = model_info["local_dir"]
        if local_dir.exists():
            file_count = len(list(local_dir.rglob("*")))
            print(f"[OK] {model_info['description']}: {local_dir} ({file_count} 个文件)")
        else:
            print(f"[FAIL] {model_info['description']}: 未下载")

    # 更新配置说明
    print("\n" + "="*60)
    print("配置说明")
    print("="*60)
    print("模型下载后，修改以下文件使用本地模型:")
    print("\n1. bert_ner.py:")
    print(f"   BERTNER(model_name=r'{MODELS_DIR / 'chinese-bert-wwm-ext'}')")
    print("\n2. sentence_encoder.py:")
    print(f"   SemanticMatcher(model_name=r'{MODELS_DIR / 'paraphrase-multilingual-MiniLM-L12-v2'}')")
    print("="*60)


if __name__ == "__main__":
    main()
