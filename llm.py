from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
import os
import sys
import argparse
from pathlib import Path
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import uuid
from typing import List, Tuple
import base64
from openai import OpenAI
import logging
import uvicorn

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag-api")

# 设置RAG文档路径
RAG_PDF_PATH = Path('/home/czy/workspace/rag.pdf')

# 初始化FastAPI应用
app = FastAPI()

# 初始化 OpenAI 客户端
client = None

def init_client():
    global client
    try:
        if client is None:
            logger.info("初始化OpenAI客户端")
            client = OpenAI(
                api_key='ollama',
                base_url='http://localhost:11434/v1/'
            )
        return client, None
    except Exception as e:
        error_msg = f"初始化客户端失败: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

# 纯文本请求
def get_llm_response(prompt, model="gemma3:4b"):
    # 获取客户端
    client, error = init_client()
    if error:
        return error
    try:
        logger.info(f"调用LLM模型: {model}")
        # 模型调用
        completion = client.chat.completions.create(
            model=model, # 模型IP，默认为gemma3:4b
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10000  # 限制token数量，获取简短回复
        )
        # 返回模型输出
        return completion.choices[0].message.content
    
    except Exception as e: #异常
        error_msg = f"LLM API调用错误: {str(e)}"
        logger.error(error_msg)
        return f"API调用失败: {str(e)}"

# 支持带图像的多模态请求
def get_multimodal_llm_response(prompt, image_path=None, model="gemma3:4b"):
    # 获取客户端
    client, error = init_client()
    if error:
        return error
    try:
        if image_path and os.path.exists(image_path):
            # 准备图像
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode()
            # 创建多模态请求
            completion = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"} # 图像数据
                    ]
                }],
                max_tokens=10000
            )
        else:
            # 创建纯文本请求
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10000
            )
        # 返回模型输出
        return completion.choices[0].message.content
    
    except Exception as e:
        error_msg = f"多模态LLM API调用错误: {str(e)}"
        logger.error(error_msg)
        return f"API调用失败: {str(e)}"

class QueryResponse(BaseModel):
    answer: str
# 处理PDF文件并进行RAG查询
def process_and_query(file_path: str, question: str) -> str: 
    try:
        logger.info("[RAG] 开始处理RAG查询...")
        logger.info(f"[RAG] 问题: {question[:50]}...")
        
        # 初始化嵌入模型
        logger.info("[RAG] 初始化嵌入模型 herald/dmeta-embedding-zh...")
        embeddings = OllamaEmbeddings(
            model="herald/dmeta-embedding-zh",
            base_url="http://localhost:11434"
        )
        
        # 加载和处理PDF
        logger.info(f"[RAG] 加载PDF文件: {file_path}")
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        logger.info(f"[RAG] PDF加载完成: {len(docs)} 页")
        
        # 文本分块
        logger.info("[RAG] 开始文本分块...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len
        )
        documents = text_splitter.split_documents(docs)
        logger.info(f"[RAG] 文本分块完成: {len(documents)} 个块")
        
        # 创建向量存储
        logger.info("[RAG] 创建向量存储...")
        vector_store = FAISS.from_documents(documents, embeddings)
        logger.info("[RAG] 向量存储创建完成")
        
        # 初始化LLM
        logger.info("[RAG] 初始化Gemma 3B...")
        llm = Ollama(
            model="gemma3:4b",
            temperature=0.3,
            base_url="http://localhost:11434"
        )
        
        # 获取相关文档
        logger.info("[RAG] 执行相似度搜索...")
        retrieved_docs: List[Tuple] = vector_store.similarity_search_with_score(question, k=3)
        for i, (doc, score) in enumerate(retrieved_docs):
            logger.info(f"[RAG] 文档 {i+1} 相似度得分: {score}")
        
        score_threshold = 0.7  # 根据实际效果调整阈值
        
        # 提取相关性达标的文档内容
        relevant_context = "\n".join([
            doc.page_content 
            for doc, score in retrieved_docs 
            if score > score_threshold
        ])
        
        # 动态构建提示词
        prompt_template = ""
        if relevant_context:
            prompt_template = f"""结合本地知识，并基于上下文使用中文回答问题：
            Context: {relevant_context}
            Question: {question}
            """
        else:
            prompt_template = f"""使用中文回答问题：
            Question: {question}
            """   
        # 生成最终答案
        logger.info("[RAG] 开始LLM推理...")
        response = llm.invoke(prompt_template)
        logger.info("[RAG] 推理完成")
        logger.info(f"[RAG] 响应内容:\n{response}")
        
        return response
    
    except Exception as e:
        error_msg = f"[RAG] 处理失败: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

@app.post("/ask/")
async def rag_endpoint(
    file: UploadFile = File(...),
    question: str = Form(...)
) -> QueryResponse:
    temp_file = None
    try:
        logger.info(f"接收到请求: 文件={file.filename}, 问题长度={len(question)}")
        # 生成临时文件名
        temp_file = f"temp_{uuid.uuid4()}.pdf"
        
        # 保存上传文件
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"已保存临时文件: {temp_file}")
        
        # 处理PDF并获取答案
        answer = process_and_query(temp_file, question)
        logger.info("成功生成回答")
        return QueryResponse(answer=answer)
        
    except Exception as e:
        error_msg = f"请求处理失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
            logger.info(f"已删除临时文件: {temp_file}")

@app.post("/ask_with_rag/")
async def ask_with_default_rag(question: str = Form(...)) -> QueryResponse:
    """使用默认RAG文档进行问答"""
    try:
        logger.info(f"接收到使用默认RAG文档的请求: 问题长度={len(question)}")
        
        # 检查RAG文档是否存在
        if not RAG_PDF_PATH.exists():
            error_msg = f"默认RAG文档不存在: {RAG_PDF_PATH}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        # 处理PDF并获取答案
        answer = process_and_query(str(RAG_PDF_PATH), question)
        logger.info("成功生成回答")
        return QueryResponse(answer=answer)
        
    except Exception as e:
        error_msg = f"请求处理失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health_check():
    """健康检查接口，用于验证服务是否正常运行"""
    try:
        # 验证Ollama服务是否可访问
        client, error = init_client()
        if error:
            return {"status": "error", "message": error}
        
        # 检查RAG文档
        if not RAG_PDF_PATH.exists():
            return {"status": "warning", "message": f"RAG文档不存在: {RAG_PDF_PATH}"}
        
        return {"status": "ok", "message": "RAG API服务正常运行"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def ensure_rag_pdf():
    """确保RAG文档存在"""
    try:
        # 确保父目录存在
        RAG_PDF_PATH.parent.mkdir(exist_ok=True)
        
        # 检查RAG文档是否存在
        if not RAG_PDF_PATH.exists():
            logger.warning(f"RAG文档不存在: {RAG_PDF_PATH}")
            logger.info("创建示例PDF文档...")
            
            # 写入简单的内容（实际上PDF需要使用specialized库创建，这里只是写入纯文本）
            with open(RAG_PDF_PATH, 'w', encoding='utf-8') as f:
                f.write("""# 通用安全处置指南
                        本文档提供通用安全事件的处置方案。
                        ## 风险评估
                        发现未知物品或可疑情况时，首先评估潜在风险。
                        ## 物品处置步骤
                        1. 保持冷静，确保个人安全
                        2. 报告相关部门并描述情况
                        3. 疏散周围人员
                        4. 等待专业人员处理
                        ## 相关部门
                        - 安保部门
                        - 应急管理中心
                        - 管理层
                        """)
            logger.info(f"示例RAG文档已创建: {RAG_PDF_PATH}")
    except Exception as e:
        logger.error(f"初始化RAG文档错误: {str(e)}")

def start_server(host="0.0.0.0", port=8008):
    """启动RAG API服务器"""
    logger.info(f"启动RAG API服务器 {host}:{port}...")
    ensure_rag_pdf()
    uvicorn.run(app, host=host, port=port)

def main():
    """主函数：解析命令行参数并启动服务器"""
    parser = argparse.ArgumentParser(description="RAG API服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8008, help="服务器端口")
    parser.add_argument("--rag-pdf", help="RAG PDF文档路径")
    
    args = parser.parse_args()
    
    # 设置RAG文档路径
    global RAG_PDF_PATH
    if args.rag_pdf:
        RAG_PDF_PATH = Path(args.rag_pdf)
        logger.info(f"使用指定的RAG文档: {RAG_PDF_PATH}")
    
    # 启动服务器
    start_server(args.host, args.port)

if __name__ == "__main__":
    main()

