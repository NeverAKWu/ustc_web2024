from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Tongyi
from getpass import getpass
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
import csv

os.environ["DASHSCOPE_API_KEY"] = "sk-001659b993184476b5f3a0984192e616"
llm = Tongyi()
print("load llm ok!")


embedding_model = HuggingFaceBgeEmbeddings(model_name="moka-ai/m3e-base")
db = FAISS.load_local("../data/faiss_index", embedding_model,  allow_dangerous_deserialization=True)


template = """你是专业的法律知识问答助手。你需要使用以下检索到的上下文片段来回答问题，禁止根据常识和已知信息回答问题。如果你不知道答案，直接回答“未找到相关答案”。
Question: {question}
Context: {context}
Answer: """

prompt = ChatPromptTemplate.from_template(template)
print("prompt generate ok!")

retriever = db.as_retriever()
rag_chain = (
{"context": retriever, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
)

answer_llm = []
answer_llm_RAG = []
answer = []
questions = ["借款人去世，继承人是否应履行偿还义务？", "如何通过法律手段应对民间借贷纠纷？", "没有赡养老人就无法继承财产吗？", "谁可以申请撤销监护人的监护资格？", "你现在是一个精通中国法律的法官，请对以下案件做出分析：经审理查明：被告人 xxx 于 2017 年 12 月，多次在本市 xxx 盗窃财物。具体事实如下：（一）2017 年 12 月 9 日 15 时许，被告人 xxx 在 xxx 店内，盗窃白色毛衣一件（价值人民币 259 元）。现赃物已起获并发还。（二）2017 年 12 月 9 日 16 时许，被告人 xx 在本市 xxx 店内，盗窃米白色大衣一件（价值人民币 1199 元）。现赃物已起获并发还。（三）2017 年 12月 11 日 19 时许，被告人 xxx 在本市 xxx 内，盗窃耳机、手套、化妆镜等商品共八件（共计价值人民币 357.3 元）。现赃物已起获并发还。（四）2017 年 12 月 11 日 20 时许，被告人 xx 在本市 xxxx 内，盗窃橙汁、牛肉干等商品共四件（共计价值人民币 58.39 元）。现赃物已起获并发还。2017 年 12 月 11 日，被告人 xx 被公安机关抓获，其到案后如实供述了上述犯罪事实。经鉴定，被告人 xxx 被诊断为精神分裂症，限制刑事责任能力，有受审能力。", "你现在是一个精通中国法律的法官，请对以下案件做出分析：2012 年 5月 1 日，原告 xxx 在被告 xxxx 购买“玉兔牌”香肠 15 包，其中价值 558.6 元的 14 包香肠已过保质期。xxx 到收银台结账后，即径直到服务台索赔，后因协商未果诉至法院，要求 xxxx 店支付 14 包香肠售价十倍的赔偿金 5586 元。"]
for question in questions:
    origin_texts = db.similarity_search(question, k=3)
    texts = [text.page_content for text in origin_texts]
    answer.append(texts)
    answer_llm.append(llm(question))
    answer_llm_RAG.append(rag_chain.invoke(question))


filename = "./question_answer.csv"

# 写入 CSV 文件
with open(filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    writer.writerow(["问题", "检索到的相关文本", "LLM回答", "LLM-RAG回答"])
    
    for i in range(len(questions)):
        writer.writerow([questions[i], "\n".join(answer[i]), answer_llm[i], answer_llm_RAG[i]])
