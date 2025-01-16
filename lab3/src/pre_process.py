from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np


# 加载CSV文件
loader = CSVLoader(file_path="../data/law_data_3k.csv", encoding='utf-8')
documents = loader.load()

# 初始化文本分割器
text_splitter = CharacterTextSplitter(
    #separator="\n",  # 分割符
    chunk_size=500,  # 每个块的字符数
    chunk_overlap=0  # 块之间的重叠部分
)

# 分割文本
splitted_documents = text_splitter.split_documents(documents)

#向量化
embedding_model = HuggingFaceBgeEmbeddings(model_name="moka-ai/m3e-base")

# 数据入库
db = FAISS.from_documents(splitted_documents, embedding_model)

db.save_local("../data/faiss_index")

