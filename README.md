# Yollm
本仓库只作为大学生创新训练计划的记录，项目程序是通过AI辅助方式构建的，**屎山警告**。如果您愿意给出一些有用的建议和优化，不胜感谢！

![image](https://github.com/user-attachments/assets/8fd44231-6913-42e7-9ba5-7669ef0a3014)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目简介

本项目实现了一套基于端侧大语言模型 (LLM) 的无人机动态环境感知与智能决策响应系统。系统通过在无人机边缘计算平台（如 Jetson AGX Xavier）部署轻量化 LLM (Gemma 3:12b)，结合本地化的实时视觉感知 (YOLOv5) 与检索增强生成 (RAG) 技术，实现了从环境感知、危险识别、基于领域知识的态势评估到生成处置方案的端侧任务闭环。该系统旨在克服传统无人机系统对云端计算和稳定网络连接的依赖，实现低延迟、高自主性的智能决策，特别适用于应急救援、灾害监测、安防巡逻等复杂和时间敏感型应用场景。

核心功能包括：
1.  **实时视觉感知模块**：利用 YOLOv5 模型在无人机端进行实时目标检测。
2.  **预警识别机制模块**：根据检测结果和预设规则，识别潜在危险目标并发出警报。
3.  **本地知识增强模块**：集成 PDF 文档作为本地知识库，通过 RAG 技术为 LLM 提供领域知识，提升响应的专业性和准确性。
4.  **LLM推理决策模块**：利用轻量级 LLM (通过 Ollama 部署) 生成针对危险情况的处置建议。
5.  **双向异构通信模块**：通过 TCP/IP 实现无人机端与地面站之间结构化文本数据（JSON 格式）和按需图像数据的低延迟传输。
6.  **可视交互界面模块**：地面站采用 Gradio 构建用户界面，实时展示检测结果、危险警报、处置方案和请求实时图像。

## 工程文件：

本项目工程主要由三个代码文件组成：

1.  **无人机端应用 (`main.py`)**:
    *   负责通过摄像头捕获视频流。
    *   运行 YOLOv5 模型进行实时目标检测。
    *   实现危险品检测逻辑，并根据历史检测结果判断是否触发警报。
    *   将检测结果、危险警报通过 TCP 发送给地面站。
    *   按需提供带有检测框的实时图像帧给地面站。
    *   当触发危险警报时，调用本地 RAG LLM API (`llm.py`) 获取处置方案，并将方案发送给地面站。
    *   自动检测并尝试启动 RAG LLM API 服务。

2.  **RAG LLM API 服务 (`llm.py`)**:
    *   一个基于 FastAPI 的 Web 服务，提供 RAG 功能。
    *   使用 Ollama 加载和运行嵌入模型 (`herald/dmeta-embedding-zh`) 和大语言模型 (`gemma3:4b`)。
    *   加载指定的 PDF 文档（默认为 `rag.pdf`），进行文本分块和向量化，构建 FAISS 向量数据库。
    *   提供 API 接口 (`/ask_with_rag/`)，接收问题，检索相关上下文，并结合 LLM 生成回答。

3.  **地面站接收与展示应用 (`receiver.py`)**:
    *   作为 TCP 服务器，接收来自无人机端的检测数据、警报信息和 LLM 响应。
    *   提供 Gradio Web UI，实时展示：
        *   检测到的物体列表。
        *   危险物品警报列表及其处理状态。
        *   选中警报后，显示 LLM 生成的处置方案（Markdown 格式化为 HTML）。
    *   允许用户请求并显示无人机端的当前实时图像。


## 环境准备与依赖

1.  **硬件**:
    *   无人机端：具备一定计算能力的边缘设备（如 NVIDIA Jetson 系列），连接有摄像头。
    *   地面站端：笔记本电脑即可。
    *   两端设备需在同一局域网内。

2.  **软件**:
    *   Python 3.8 或更高版本。
    *   **Ollama**: 必须安装并运行。
        *   拉取所需模型：
            ```bash
            ollama pull gemma3:4b
            ollama pull herald/dmeta-embedding-zh
            ```
        *   确保 Ollama 服务正在运行（通常在 `http://localhost:11434`）。
    *   CUDA (可选，但强烈推荐在无人机端使用以加速 YOLOv5 和可能的 LLM 推理)。

3.  **Python 依赖**:
    克隆本仓库后，在项目根目录下创建 `requirements.txt` 文件，内容如下：
    ```txt
    fastapi
    uvicorn
    pydantic
    python-multipart
    langchain-community
    langchain
    ollama
    faiss-gpu
    pdfplumber
    openai
    gradio
    pandas
    numpy
    opencv-python
    torch
    torchvision # 根据你的YOLOv5版本和PyTorch版本选择合适的
    requests
    # YOLOv5 本身可能需要特定版本的torch和torchvision，请根据YOLOv5官方指引安装
    ```
    然后安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

4.  **YOLOv5 模型**:
    *   在 `main.py` 中，`yolov5_path` 和 `weights_path` 指向您本地 YOLOv5 代码库和模型权重文件的路径。请确保这些路径正确。
        ```python
        # main.py
        yolov5_path = Path('/home/czy/workspace/yolov5-7.0')  # 修改为你的YOLOv5 v7.0目录路径
        weights_path = yolov5_path / 'yolov5m.pt'  # 模型权重路径
        ```
    *   如果尚未下载 YOLOv5，请从官方仓库克隆并下载预训练权重（如 `yolov5m.pt`）。

5.  **RAG 文档**:
    *   默认的 RAG PDF 文档路径在 `llm.py` 和 `main.py` (用于启动`llm.py`时传递参数) 中定义为 `RAG_PDF_PATH = Path('/home/czy/workspace/rag.pdf')`。
    *   如果该文件不存在，系统会尝试创建一个简单的示例文本文件并命名为 `rag.pdf`。您可以替换为您自己的 PDF 知识库文件。
    *   确保 `llm.py` 和 `main.py` 中的 `RAG_PDF_PATH` 指向同一个有效文件，或者在启动时通过命令行参数指定。

6.  **网络配置**:
    *   **IP 地址**: 代码中硬编码了 IP 地址。您需要根据您的实际网络环境修改这些地址：
        *   `main.py`:
            *   `message_socket.connect(('192.168.3.153', 5000))`: 地面站 `receiver.py` 运行的 IP 和端口。
            *   `server_socket.bind(('192.168.3.146', 5001))`: 无人机端图像服务绑定的 IP（应为无人机自身的 IP）。
            *   `RAG_API_URL = "http://localhost:8008"`: RAG LLM API 服务地址。如果 `llm.py` 和 `main.py` 在同一设备运行，`localhost` 通常可行。
        *   `receiver.py`:
            *   `self.sock.bind(('192.168.3.153', port))`: 地面站消息服务绑定的 IP (应为地面站自身的 IP)。
            *   `image_socket.connect(('192.168.3.146', 5001))`: 无人机端图像服务的 IP 和端口。
        *   `llm.py`:
            *   `base_url='http://localhost:11434/v1/'`: Ollama 服务地址。
            *   默认 FastAPI 服务运行在 `0.0.0.0:8008`。
    *   **重要提示**: 确保所有组件（无人机、地面站、Ollama服务）之间的网络是互通的，并且防火墙设置允许相应的 TCP 通信。

## 运行步骤

请按以下顺序启动各个组件：

1.  **启动 Ollama 服务** (如果尚未运行):
    通常 Ollama 安装后会自动作为后台服务运行。您可以通过 `ollama list` 检查。

2.  **启动 RAG LLM API 服务 (`llm.py`)**:
    (如果 `main.py` 配置为自动启动此服务，此步骤可能不是严格必需的，但手动启动有助于调试)
    在无人机端（或运行 LLM 服务的机器上）执行：
    ```bash
    python llm.py --host 0.0.0.0 --port 8008 --rag-pdf /path/to/your/rag.pdf
    ```
    *   `--host`: API 绑定的主机地址。
    *   `--port`: API 绑定的端口。
    *   `--rag-pdf`: (可选) 指定 RAG 使用的 PDF 文件路径，如果与代码中默认值不同。
    *   服务启动后，可以通过访问 `http://<llm_server_ip>:8008/health` 来检查其健康状况。

3.  **启动地面站接收与展示应用 (`receiver.py`)**:
    在地面站电脑上执行：
    ```bash
    python receiver.py
    ```
    *   该脚本会启动一个 TCP 服务器等待无人机连接，并启动一个 Gradio Web UI。
    *   留意控制台输出的 Gradio UI 地址，通常是 `http://<ground_station_ip>:7860`。在浏览器中打开此地址。

4.  **启动无人机端应用 (`main.py`)**:
    在无人机边缘计算设备上执行：
    ```bash
    python main.py
    ```
    *   该脚本会尝试连接到地面站的 TCP 服务器和 RAG LLM API。
    *   如果 RAG LLM API 未运行，它会尝试在本地将其作为子进程启动（依赖 `llm.py` 和正确的 Python 环境）。
    *   启动后，它会开始进行目标检测，并将数据发送到地面站。
    *   它也会启动一个 Gradio 界面用于显示实时视频流，通常在 `http://<drone_ip>:7860` (端口可能与地面站UI冲突，但通常 `main.py`的Gradio用于本地调试，主要交互通过`receiver.py`的UI)。

**运行流程概述**:
*   `receiver.py` 先启动，监听特定IP和端口的消息。
*   `main.py` (无人机端) 启动后，连接到 `receiver.py` 设定的IP和端口，开始发送检测数据。
*   如果 `main.py` 检测到危险物品，它会调用 (本地或远程的) `llm.py` API 服务。
*   `llm.py` 服务处理请求，利用 RAG 和 LLM 生成响应。
*   `main.py` 将 LLM 响应发送给 `receiver.py`。
*   用户在 `receiver.py` 的 Gradio 界面上查看所有信息。
