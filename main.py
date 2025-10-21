import torch
import cv2
from pathlib import Path
from datetime import datetime
import gradio as gr
import socket
import pickle
import struct
import time
import numpy as np
import threading
import json
import base64
import requests
import os
import subprocess
import sys
import signal

# 设置YOLOv5模型路径和权重
yolov5_path = Path('/home/czy/workspace/yolov5-7.0')  # 本地yolov5 v7.0目录路径
weights_path = yolov5_path / 'yolov5m.pt'  # 模型权重路径

# 定义危险物品类别和标签
DANGER_CATEGORIES = {
        "Animal": ["person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
        "Sharp Objects": ["fork", "knife", "scissors"],
        "Vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
}

# 反向映射，用于快速查找标签所属类别
LABEL_TO_CATEGORY = {}
for category, labels in DANGER_CATEGORIES.items():
    for label in labels:
        LABEL_TO_CATEGORY[label] = category

# 危险物品检测配置
DANGER_DETECTION_THRESHOLD = 0.6  # 置信度阈值
DANGER_DETECTION_WINDOW = 3  # 检测窗口（秒）
DANGER_DETECTION_COUNT = 3  # 检测次数阈值

# LLM请求状态跟踪
llm_request_status = {}  # 格式: {label: {"in_progress": bool, "llm_sent": bool}}
llm_request_lock = threading.Lock()  # 用于同步LLM请求状态的锁

# RAG配置
RAG_API_URL = "http://localhost:8008"  # RAG API服务器地址
RAG_PDF_PATH = Path('/home/czy/workspace/rag.pdf')  # RAG PDF文档路径
RAG_API_PROCESS = None  # 用于存储启动的API进程

# 测试模式标志
TEST_MODE = False
# 测试响应延迟(秒)
TEST_RESPONSE_DELAY = 5

# 确保RAG PDF文档存在
def ensure_rag_pdf():
    """确保RAG文档存在"""
    try:
        # 确保父目录存在
        RAG_PDF_PATH.parent.mkdir(exist_ok=True)
        
        # 检查RAG文档是否存在
        if not RAG_PDF_PATH.exists():
            print(f"RAG文档不存在: {RAG_PDF_PATH}")
            print("创建示例PDF文档...")
            
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
            print(f"示例RAG文档已创建: {RAG_PDF_PATH}")
    except Exception as e:
        print(f"初始化RAG文档错误: {str(e)}")

# 检查RAG API服务是否运行
def is_rag_api_running():
    """检查RAG API服务是否运行"""
    try:
        response = requests.get(f"{RAG_API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# 启动RAG API服务
def start_rag_api():
    """启动RAG API服务"""
    global RAG_API_PROCESS
    
    print("[RAG] 正在启动RAG API服务...")
    
    try:
        # 检查llm.py文件是否存在
        llm_path = Path('/home/czy/workspace/mi/llm.py')
        if not llm_path.exists():
            print(f"[RAG] 错误: 找不到llm.py文件: {llm_path}")
            return False
        
        print(f"[RAG] 使用Python解释器: {sys.executable}")
        print(f"[RAG] 启动llm.py: {llm_path}")
        
        # 使用subprocess在后台启动服务
        RAG_API_PROCESS = subprocess.Popen(
            [sys.executable, str(llm_path), "--host", "0.0.0.0", "--port", "8008", "--rag-pdf", str(RAG_PDF_PATH)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # 使用新进程组
        )
        
        # 等待服务启动（最多等待30秒）
        timeout = 30
        start_time = time.time()
        while not is_rag_api_running():
            if time.time() - start_time > timeout:
                print(f"[RAG] 启动RAG API服务超时（{timeout}秒）")
                if RAG_API_PROCESS.poll() is not None:
                    print(f"[RAG] 进程输出:\n{RAG_API_PROCESS.stdout.read()}")
                    print(f"[RAG] 错误输出:\n{RAG_API_PROCESS.stderr.read()}")
                return False
            
            # 检查进程是否仍在运行
            if RAG_API_PROCESS.poll() is not None:
                print("[RAG] RAG API服务启动失败")
                print(f"[RAG] 进程输出:\n{RAG_API_PROCESS.stdout.read()}")
                print(f"[RAG] 错误输出:\n{RAG_API_PROCESS.stderr.read()}")
                return False
                
            print("[RAG] 等待RAG API服务启动...")
            time.sleep(2)
        
        print("[RAG] RAG API服务已成功启动!")
        return True
    
    except Exception as e:
        print(f"[RAG] 启动RAG API服务时出错: {str(e)}")
        return False

# 确保RAG文档存在并启动API服务
ensure_rag_pdf()
if not is_rag_api_running():
    start_rag_api()

# 从llm.py调用API获取LLM响应
def get_llm_response(prompt, model="gemma3:4b"):
    try:
        # 检查服务健康状态
        response = requests.get(f"{RAG_API_URL}/health")
        if response.status_code != 200:
            print(f"RAG API服务未正常运行: {response.text}")
            
            # 尝试重启服务
            if not is_rag_api_running() and not start_rag_api():
                return f"RAG API服务未正常运行且无法启动"
            
            # 重试请求
            response = requests.get(f"{RAG_API_URL}/health")
            if response.status_code != 200:
                return f"RAG API服务无法正常响应: {response.text}"
        
        # 使用默认RAG文档查询
        response = requests.post(
            f"{RAG_API_URL}/ask_with_rag/",
            data={"question": prompt}
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("answer", "获取响应失败")
        else:
            error_msg = f"API调用失败: HTTP {response.status_code}"
            print(error_msg)
            return error_msg
    except Exception as e:
        error_msg = f"LLM API调用错误: {str(e)}"
        print(error_msg)
        return error_msg

# 加载本地模型
try:
    model = torch.hub.load(str(yolov5_path),
                      'custom',
                      path=weights_path,
                      source='local',
                      force_reload=True)
    print("YOLOv5模型加载成功")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    # 使用一个简单的测试模型
    TEST_MODE = True
    print("启用测试模式")

# 设置视频源
if not TEST_MODE:
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, test_frame = cap.read()
        if not ret:
            print("摄像头初始化失败，启用测试模式")
            TEST_MODE = True
    except Exception as e:
        print(f"摄像头初始化错误: {str(e)}")
        TEST_MODE = True

# 全局计数器和输出文件
detection_count = 0
output_file = open('detections.txt', 'w')
output_file.write("时间戳,标签,置信度\n")

# 创建消息套接字
message_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    print("连接到消息服务器...")
    message_socket.connect(('192.168.3.153', 5000))
    print("消息服务器连接成功")
except Exception as e:
    print(f"连接消息服务器失败: {str(e)}")
    print("将在消息发送时重试连接")

# 当前帧与检测结果缓存
current_frame = None
current_result_frame = None
frame_lock = threading.Lock()

# 危险物品检测历史记录
danger_history = {}  # 格式: {label: [(timestamp, confidence), ...]}

# 监听来自客户端的图像请求，并发送当前检测结果图像
def image_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind(('192.168.3.146', 5001))
        server_socket.listen(1)
        print("图像服务器启动在端口5001")
        
        while True:
            print("等待图像请求连接...")
            client_socket, addr = server_socket.accept()
            print(f"接收到图像请求连接: {addr}")
            
            try:
                # 接收请求命令
                request = client_socket.recv(1024).decode('utf-8')
                if request == "REQUEST_IMAGE":
                    print("收到图像请求")
                    
                    # 获取当前带标签的检测结果帧
                    with frame_lock:
                        if current_result_frame is not None:
                            frame = current_result_frame.copy()
                        else:
                            print("没有可用的检测结果图像")
                            client_socket.close()
                            continue
                    
                    # 压缩图像
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    
                    # 发送图像数据大小
                    size = len(buffer)
                    size_data = struct.pack("!I", size)
                    client_socket.sendall(size_data)
                    
                    # 发送图像数据
                    client_socket.sendall(buffer)
                    print(f"图像发送完成，大小: {size}字节")
            except Exception as e:
                print(f"处理图像请求出错: {str(e)}")
            finally:
                client_socket.close()
    except Exception as e:
        print(f"图像服务器错误: {str(e)}")
    finally:
        server_socket.close()

# 启动图像服务器线程
image_thread = threading.Thread(target=image_server, daemon=True)
image_thread.start()

# 检查是否为危险物品
def is_danger_item(label):
    for category, labels in DANGER_CATEGORIES.items():
        if label.lower() in labels:
            return True
    return False

# 获取物品的类别
def get_item_category(label):
    return LABEL_TO_CATEGORY.get(label.lower(), "未知类别")

# 检查危险物品是否达到警报阈值
def check_danger_threshold(label, confidence, current_time):
    global danger_history
    
    # 如果不是危险物品，直接返回False
    if not is_danger_item(label):
        return False
    
    # 添加当前检测到历史记录
    if label not in danger_history:
        danger_history[label] = []
    
    # 添加当前检测结果
    danger_history[label].append((current_time, confidence))
    
    # 清理超出时间窗口的记录
    window_start = current_time - DANGER_DETECTION_WINDOW
    danger_history[label] = [(t, c) for t, c in danger_history[label] if t >= window_start]
    
    # 检查置信度大于阈值的检测次数
    high_confidence_count = sum(1 for _, c in danger_history[label] if c >= DANGER_DETECTION_THRESHOLD)
    
    # 如果次数达到阈值，触发警报
    return high_confidence_count >= DANGER_DETECTION_COUNT

# 发送消息，带有自动重连功能
def send_message(message):
    global message_socket
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            message_socket.send(message.encode('utf-8'))
            return True
        except Exception as e:
            retry_count += 1
            print(f"发送消息失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
            
            # 尝试重新连接
            try:
                message_socket.close()
                message_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                message_socket.connect(('192.168.3.153', 5000))
                print("重新连接到消息服务器成功")
            except Exception as conn_err:
                print(f"重新连接失败: {str(conn_err)}")
                time.sleep(1)  # 等待一秒后重试
    
    print("发送消息失败，达到最大重试次数")
    return False

# 发送带有特殊结束标记的长文本消息
def send_long_message(message):
    # 添加消息结束标记
    message_with_end = message + "\n<<END_OF_MESSAGE>>\n"
    try:
        # 分块发送大消息，确保完整传输
        chunk_size = 4096
        for i in range(0, len(message_with_end), chunk_size):
            chunk = message_with_end[i:i+chunk_size]
            result = send_message(chunk)
            if not result:
                return False
            # 添加短暂延迟避免消息粘连
            if i + chunk_size < len(message_with_end):
                time.sleep(0.01)
        return True
    except Exception as e:
        print(f"发送长消息失败: {str(e)}")
        return False

# 发送结构化JSON消息
def send_json_message(msg_type, content_dict):
    """
    参数:
        msg_type: 消息类型 ("DETECTION", "DANGER_ALERT", "LLM_RESPONSE")
        content_dict: 包含消息内容的字典
    """
    # 创建完整消息
    message = {
        "type": msg_type,
        "timestamp": datetime.now().isoformat(),
        "content": content_dict
    }
    
    # 转换为JSON字符串
    json_str = json.dumps(message)
    
    # 编码并发送
    return send_long_message(json_str)

# 检查是否需要为该标签发送危险警报
def should_send_danger_alert(label):
    with llm_request_lock:
        # 如果标签不在状态字典中，需要发送警报
        if label not in llm_request_status:
            return True
            
        status = llm_request_status[label]
        current_time = time.time()
        
        # 如果警报已发送但推理仍在进行中，检查是否超时
        if status.get("alert_sent", False) and status.get("in_progress", False):
            last_time = status.get("last_request_time", 0)
            # 超时处理 - 100秒后可以重新发送
            if current_time - last_time > 100:
                print(f"LLM请求超时: {label}, 重置状态")
                return True
            return False
            
        # 如果LLM响应已经发送，检查是否过了冷却期（5分钟）
        if status.get("llm_sent", False):
            last_time = status.get("last_request_time", 0)
            if current_time - last_time > 300:  # 5分钟后可以重新发起警报
                return True
            return False
            
        # 其他情况下可以发送警报
        return True

# 发送危险警报
def send_danger_alert(label, confidence, current_time):
    # 检查是否应该发送警报
    if not should_send_danger_alert(label):
        print(f"已为 {label} 发送警报或LLM正在处理")
        return False
        
    try:
        # 发送警报信息
        category = get_item_category(label)
        alert_data = {
            "label": label,
            "category": category,
            "confidence": float(confidence),
            "datetime": str(current_time)
        }
        
        if send_json_message("DANGER_ALERT", alert_data):
            print(f"已发送危险警报: {label}")
            
            # 更新状态 - 标记警报已发送，开始LLM处理
            with llm_request_lock:
                llm_request_status[label] = {
                    "in_progress": True,
                    "alert_sent": True,
                    "llm_sent": False,
                    "last_request_time": time.time()
                }
            
            # 启动异步LLM响应生成
            llm_thread = threading.Thread(
                target=generate_llm_response_async, 
                args=(label, confidence, current_time)
            )
            llm_thread.daemon = True
            llm_thread.start()
            
            return True
        return False
    except Exception as e:
        print(f"发送危险警报失败: {str(e)}")
        return False

# 生成LLM提示并获取响应 (异步处理)
def generate_llm_response_async(label, confidence, current_time):
    try:
        print(f"[LLM] 开始为 {label} 生成LLM响应...")
        print(f"[LLM] 验证RAG API服务状态...")
        
        # 验证RAG API服务是否正常运行
        if not is_rag_api_running():
            print("[LLM] RAG API服务未运行，尝试重新启动...")
            if not start_rag_api():
                raise Exception("无法启动RAG API服务")
        
        # 确认警报已经发送
        with llm_request_lock:
            if label not in llm_request_status or not llm_request_status[label].get("alert_sent", False):
                print(f"警报未发送，不进行LLM处理: {label}")
                return
        
        category = get_item_category(label)
        
        # 构建提示模板
        print(f"[LLM] 构建提示模板: {category}类的{label}")
        prompt = f"""在该受监管环境中，出现了{category}类的{label}，置信度为{confidence:.2f}，属于危险物品。
                    请按照该类危险品的预备方案规划，结合当前情况给出可行解决方案。
                    方案应该简明扼要，包括：
                    1. 风险评估
                    2. 紧急处置步骤
                    3. 需要通知的相关部门
                    """
        
        print(f"[LLM] 开始LLM推理...")
        response_start_time = time.time()
        
        # 测试模式
        if TEST_MODE:
            print(f"[LLM] 测试模式: 模拟{TEST_RESPONSE_DELAY}秒推理延迟")
            time.sleep(TEST_RESPONSE_DELAY)
            response = f"""测试响应 - {category}类的{label}危险物品处置方案:
                        1. 风险评估：发现{label}可能带来安全隐患，需立即处理。
                        2. 紧急处置：疏散周围人员，保持安全距离，专业人员处理。
                        3. 通知部门：安保部门、紧急应对小组、管理层。
                        """
        else:
            # 调用LLM获取响应
            response = get_llm_response(prompt, model="gemma3:4b")
        
        response_end_time = time.time()
        print(f"[LLM] 推理完成，用时: {response_end_time - response_start_time:.2f}秒")
        print(f"[LLM] 响应内容:\n{response}")  # 打印完整响应
        
        # 发送LLM响应（JSON格式）
        llm_data = {
            "label": label,
            "category": category,
            "confidence": float(confidence),
            "response": response,
            "inference_time": round(response_end_time - response_start_time, 2)
        }
        
        # 尝试多次发送，确保成功
        max_retries = 3
        for retry in range(max_retries):
            print(f"[LLM] 尝试发送LLM响应 (尝试 {retry+1}/{max_retries})...")
            if send_json_message("LLM_RESPONSE", llm_data):
                print(f"[LLM] 已成功发送LLM响应，威胁物: {label}")
                
                # 更新状态 - 标记LLM响应已发送
                with llm_request_lock:
                    llm_request_status[label] = {
                        "in_progress": False,
                        "alert_sent": True,
                        "llm_sent": True,
                        "last_request_time": time.time()
                    }
                return
            else:
                print(f"[LLM] 第 {retry+1} 次发送LLM响应失败，等待重试...")
                time.sleep(1)  # 等待1秒后重试
                
        print(f"[LLM] 发送LLM响应失败: {label}，达到最大重试次数")
    
    except Exception as e:
        print(f"[LLM] 生成LLM响应过程出错: {str(e)}")
        
        # 出错时更新状态，允许重新尝试
        with llm_request_lock:
            if label in llm_request_status:
                llm_request_status[label]["in_progress"] = False

# 测试LLM响应函数
def test_llm_response():
    print("开始测试LLM响应...")
    test_labels = ["knife", "person", "fire"]
    
    for label in test_labels:
        confidence = 0.85
        current_time = datetime.now()
        category = get_item_category(label)
        print(f"测试 {category}类的{label}")
        
        # 发送警报并触发LLM响应
        if send_danger_alert(label, confidence, current_time):
            print(f"已触发 {label} 的测试警报")
        else:
            print(f"触发 {label} 的测试警报失败")
        
        # 等待3秒后测试下一个
        time.sleep(3)
    
    print("LLM响应测试已启动，等待LLM推理完成...")

# 清理资源
def cleanup():
    """清理资源，特别是关闭RAG API进程"""
    global RAG_API_PROCESS
    
    print("清理资源...")
    
    # 关闭RAG API进程
    if RAG_API_PROCESS:
        try:
            # 使用进程组ID来终止整个进程组
            os.killpg(os.getpgid(RAG_API_PROCESS.pid), signal.SIGTERM)
            print("已终止RAG API服务")
        except Exception as e:
            print(f"终止RAG API服务出错: {str(e)}")

# 视频检测函数
def video_detection():
    global detection_count, current_frame, current_result_frame
    last_send_time = time.time()
    danger_alerts_sent = set()  # 已发送警报的标签集合，防止重复发送
    
    # 测试模式
    if TEST_MODE:
        # 创建测试图像
        test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(test_img, "TEST MODE - NO CAMERA", (50, 360), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # 保存当前帧
        with frame_lock:
            current_frame = test_img.copy()
            current_result_frame = test_img.copy()
        
        # 运行测试
        test_thread = threading.Thread(target=test_llm_response)
        test_thread.daemon = True
        test_thread.start()
        
        while True:
            # 返回测试图像
            time.sleep(0.1)
            yield cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    # 常规模式
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取摄像头画面")
            time.sleep(0.1)
            continue
        
        # 保存当前帧
        with frame_lock:
            current_frame = frame.copy()
            
        # 转换颜色空间并进行检测
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame)
        detections = results.xyxy[0].cpu().numpy()

        # 获取带标签的渲染图像
        rendered_img = results.render()[0]  # 获取渲染后的图像（带有标签）
        
        # 将渲染后的RGB图像转回BGR用于OpenCV操作
        rendered_bgr = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
        
        # 保存带标签的检测结果帧
        with frame_lock:
            current_result_frame = rendered_bgr.copy()

        # 当前时间
        current_time = datetime.now()
        current_time_sec = time.time()

        # 写入检测结果文件并检查危险物品
        for det in detections:
            label = model.names[int(det[5])]
            confidence = det[4]
            
            # 写入文件
            output_file.write(f"{current_time},{label},{confidence:.4f}\n")
            detection_count += 1
            
            # 每10000条重置文件
            if detection_count >= 10000:
                detection_count = 0
                output_file.seek(0)
                output_file.truncate()
                output_file.write("时间戳,标签,置信度\n")
            output_file.flush()
            
            # 检查是否为危险物品并达到警报阈值
            if check_danger_threshold(label, confidence, current_time_sec):
                # 生成警报ID，防止短时间内重复发送同一警报
                alert_id = f"{label}_{int(current_time_sec/30)}"  # 每30秒可以重新发送同一物品的警报
                
                if alert_id not in danger_alerts_sent:
                    if send_danger_alert(label, confidence, current_time):
                        danger_alerts_sent.add(alert_id)
                        # 限制警报集合大小，防止内存泄漏
                        if len(danger_alerts_sent) > 100:
                            danger_alerts_sent.clear()

        # 限制发送频率，每200ms发送一次检测结果
        if current_time_sec - last_send_time > 0.2 and len(detections) > 0:
            # 发送检测结果
            try:
                for det in detections:
                    label = model.names[int(det[5])]
                    confidence = det[4]
                    
                    # 构建检测结果JSON
                    detection_data = {
                        "label": label,
                        "confidence": float(confidence),
                        "datetime": str(current_time)
                    }
                    
                    # 发送检测结果
                    send_json_message("DETECTION", detection_data)
                    time.sleep(0.01)  # 防止消息粘连
                
                last_send_time = current_time_sec
            except Exception as e:
                print(f"发送检测结果失败: {str(e)}")

        # 返回渲染后的图像
        yield rendered_img

# 创建Gradio界面
demo = gr.Interface(
    video_detection,
    inputs=None,
    outputs=gr.Image(streaming=True),
    title="基于端侧LLM的无人机动态环境决策响应系统",
    description="YOLOv5实时视觉感知&端侧LLM推理决策认知，生成危险物品本地处置方案",
    live=True
)

if __name__ == "__main__":
    try:
        print("启动基于端侧LLM的无人机动态环境决策响应系统...")
        # 测试模式下，直接运行测试
        if TEST_MODE:
            print("系统以测试模式运行，使用模拟数据")
        
        # 启动Gradio界面
        demo.queue().launch(server_name="192.168.3.146",
                          server_port=7860,
                          share=True)
    finally:
        print("关闭连接...")
        # 清理资源
        cleanup()
        message_socket.close()
        if not TEST_MODE:
            cap.release()
        cv2.destroyAllWindows()
        output_file.close()
        print("程序已关闭")
