from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModel, AutoTokenizer

# 1. 直接下载模型
model_name = "BAAI/bge-large-zh-v1.5"

# 下载模型和分词器
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 创建 embeddings 对象
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,  # 只保留 model_name 参数
    # 移除 cache_dir 参数
)
