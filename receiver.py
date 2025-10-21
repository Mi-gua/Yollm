import socket
import gradio as gr
import threading
import time
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import struct
import json

class Receiver:
    """
    TCP消息接收器类
    """
    def __init__(self):
        self.running = True  # 服务器运行状态标志
        self.current_image = None  # 当前接收到的图像
        self.lock = threading.Lock()  # 添加线程锁
        self.df = pd.DataFrame(columns=['时间', '物体', '置信度'])  # 创建数据表格
        
        # 危险物品警报和LLM响应
        self.danger_alerts = pd.DataFrame(columns=['时间', '物体', '置信度', '状态'])
        self.llm_responses = {}  # 存储LLM响应: {标签: 响应内容}
        
        # 最新收到响应的标签
        self.latest_response_label = None
        
        # 用于通知UI更新的事件
        self.llm_response_updated = threading.Event()
        
        # 消息缓冲区
        self.buffer = ""
        # 消息结束标记
        self.MSG_END_MARKER = "<<END_OF_MESSAGE>>"
        
        # 物体分类
        self.categories = {
            "Animal": ["person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
            "Sharp Objects": ["fork", "knife", "scissors"],
            "Vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
        }
        
    def start_server(self, port):
        """启动TCP服务器接收消息"""
        while self.running:
            try:
                # 创建TCP套接字
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.sock.bind(('192.168.3.153', port))  # 修改为接收端IP
                self.sock.listen(1)
                print(f"消息服务器启动在端口 {port}，等待连接...")
                self.conn, self.addr = self.sock.accept()
                print(f"消息服务连接成功，来自: {self.addr}")
                
                while self.running:
                    try:
                        # 读取数据
                        data = self.conn.recv(8192)  # 增大接收缓冲区
                        if not data:
                            continue
                        
                        # 尝试解码为文本
                        try:
                            message = data.decode('utf-8')
                            self.buffer += message
                            
                            # 处理完整消息
                            self.process_buffer()
                            
                        except UnicodeDecodeError:
                            # 忽略二进制数据
                            self.buffer = ""
                            pass
                    except Exception as e:
                        if "Connection reset by peer" in str(e):
                            print("连接断开，等待重新连接...")
                            break
                        print(f"接收消息错误: {str(e)}")
                        self.buffer = ""  # 清空缓冲区
                        time.sleep(0.1)
                
                # 连接断开，关闭套接字
                self.sock.close()
            except Exception as e:
                print(f"服务器错误: {str(e)}")
                time.sleep(1)
    
    def process_buffer(self):
        """处理缓冲区中的数据，提取完整消息"""
        if len(self.buffer) > 100000:  # 如果缓冲区超过100KB
            print(f"缓冲区过大 ({len(self.buffer)} 字节)，进行重置")
            # 保留最后10000个字符
            self.buffer = self.buffer[-10000:]
        
        # 使用结束标记分割消息
        while self.MSG_END_MARKER in self.buffer:
            try:
                # 获取完整消息
                end_pos = self.buffer.find(self.MSG_END_MARKER)
                complete_message = self.buffer[:end_pos].strip()
                # 更新缓冲区，移除已处理的消息
                self.buffer = self.buffer[end_pos + len(self.MSG_END_MARKER):]
                
                # 处理完整消息
                if complete_message:
                    self.parse_message(complete_message)
            except Exception as e:
                print(f"处理消息缓冲区出错: {str(e)}")
                # 出错时尝试移动到下一个标记
                next_marker = self.buffer.find(self.MSG_END_MARKER, 1)
                if next_marker != -1:
                    self.buffer = self.buffer[next_marker + len(self.MSG_END_MARKER):]
                else:
                    # 没有找到下一个标记，清空缓冲区
                    self.buffer = ""
                break
    
    def parse_message(self, message):
        """解析JSON格式消息"""
        try:
            # 尝试解析为JSON
            data = json.loads(message)
            
            # 根据消息类型处理
            msg_type = data.get("type", "")
            if msg_type == "DETECTION":
                self.process_detection_json(data)
            elif msg_type == "DANGER_ALERT":
                self.process_danger_alert_json(data)
            elif msg_type == "LLM_RESPONSE":
                self.process_llm_response_json(data)
            else:
                print(f"未知消息类型: {msg_type}")
                
        except json.JSONDecodeError:
            # 不是JSON格式，尝试旧格式解析
            print("无法解析JSON消息，尝试旧格式")
            if message.startswith("DANGER_ALERT,"):
                self.process_danger_alert(message)
            elif message.startswith("LLM_RESPONSE,"):
                self.process_llm_response(message)
            else:
                self.process_regular_message(message)
        except Exception as e:
            print(f"解析消息出错: {str(e)}")
    
    def process_detection_json(self, data):
        """处理检测结果JSON消息"""
        try:
            content = data.get("content", {})
            label = content.get("label", "")
            confidence = content.get("confidence", 0.0)
            timestamp = content.get("datetime", datetime.now().isoformat())
            
            # 查找物体分类
            category = "未知类别"
            for cat_name, items in self.categories.items():
                if label.lower() in [item.lower() for item in items]:
                    category = cat_name
                    break
            
            # 解析时间戳，只保留时分秒
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%H:%M:%S")
            except:
                # 如果解析失败，使用原始时间戳
                time_str = timestamp
            
            # 创建新行并添加到数据表
            with self.lock:
                new_row = pd.DataFrame({
                    '时间': [time_str],
                    '物体': [label],
                    '置信度': [float(confidence)]
                })
                self.df = pd.concat([self.df, new_row], ignore_index=True)
                # 只保留最新的20条记录
                if len(self.df) > 20:
                    self.df = self.df.iloc[-20:]
        except Exception as e:
            print(f"处理检测结果JSON出错: {str(e)}")
    
    def process_danger_alert_json(self, data):
        """处理危险警报JSON消息"""
        try:
            content = data.get("content", {})
            label = content.get("label", "")
            confidence = content.get("confidence", 0.0)
            timestamp = content.get("datetime", datetime.now().isoformat())
            
            # 查找物体分类
            category = "未知类别"
            for cat_name, items in self.categories.items():
                if label.lower() in [item.lower() for item in items]:
                    category = cat_name
                    break
            
            # 解析时间戳，只保留时分秒
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%H:%M:%S")
            except:
                # 如果解析失败，使用原始时间戳
                time_str = timestamp
            
            print(f"收到危险物品警报: {label} ({category}) 置信度: {confidence}")
            
            # 创建新警报行并添加到表格
            with self.lock:
                new_alert = pd.DataFrame({
                    '时间': [time_str],
                    '物体': [label],
                    '置信度': [float(confidence)],
                    '状态': ['等待LLM处理']
                })
                
                # 检查是否已有相同标签的警报
                existing_idx = self.danger_alerts.index[self.danger_alerts['物体'] == label].tolist()
                if existing_idx:
                    # 如果已有相同标签的警报且状态为"等待LLM处理"，则更新
                    for idx in existing_idx:
                        if self.danger_alerts.loc[idx, '状态'] == '等待LLM处理':
                            # 更新时间和置信度
                            self.danger_alerts.loc[idx, '时间'] = time_str
                            self.danger_alerts.loc[idx, '置信度'] = float(confidence)
                            break
                    else:
                        # 所有相同标签的警报都已处理，添加新警报
                        if len(existing_idx) < 2:  # 限制同一标签的警报数量
                            self.danger_alerts = pd.concat([self.danger_alerts, new_alert], ignore_index=True)
                else:
                    # 没有相同标签的警报，直接添加
                    self.danger_alerts = pd.concat([self.danger_alerts, new_alert], ignore_index=True)
                
                # 只保留最新的50条警报
                if len(self.danger_alerts) > 50:
                    self.danger_alerts = self.danger_alerts.iloc[-50:]
        except Exception as e:
            print(f"处理危险警报JSON出错: {str(e)}")
    
    def process_llm_response_json(self, data):
        """处理LLM响应JSON消息"""
        try:
            content = data.get("content", {})
            label = content.get("label", "")
            response = content.get("response", "")
            category = content.get("category", "未知类别")
            inference_time = content.get("inference_time", 0)
            
            print(f"收到LLM响应，物品: {label}，推理时间: {inference_time}秒")
            
            # 存储LLM响应
            with self.lock:
                self.llm_responses[label] = response
                print(f"已存储LLM响应: {label}, 长度: {len(response)}")
                
                # 更新对应警报的状态
                mask = self.danger_alerts['物体'] == label
                self.danger_alerts.loc[mask, '状态'] = '已处理'
                
                # 记录最新收到响应的标签，用于自动选择
                self.latest_response_label = label
                
            # 通知UI可以更新了
            self.llm_response_updated.set()
            
        except Exception as e:
            print(f"处理LLM响应JSON出错: {str(e)}")
    
    def process_regular_message(self, message):
        """处理常规检测消息（旧格式兼容）"""
        parts = message.strip().split(',')
        if len(parts) == 3:
            try:
                timestamp, label, confidence = parts
                with self.lock:
                    # 解析时间戳，只保留时分秒
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%H:%M:%S")
                    except:
                        time_str = timestamp
                    
                    # 创建新行并添加
                    new_row = pd.DataFrame({
                        '时间': [time_str],
                        '物体': [label],
                        '置信度': [float(confidence)]
                    })
                    self.df = pd.concat([self.df, new_row], ignore_index=True)
                    if len(self.df) > 20:
                        self.df = self.df.iloc[-20:]
            except Exception as e:
                print(f"解析常规消息失败: {str(e)}")
    
    def process_danger_alert(self, message):
        """处理危险物品警报消息（旧格式兼容）"""
        parts = message.strip().split(',', 3)
        if len(parts) == 4:
            try:
                _, timestamp, label, confidence = parts
                with self.lock:
                    # 解析时间戳
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%H:%M:%S")
                    except:
                        time_str = timestamp
                    
                    # 创建新警报
                    new_alert = pd.DataFrame({
                        '时间': [time_str],
                        '物体': [label],
                        '置信度': [float(confidence)],
                        '状态': ['等待LLM处理']
                    })
                    
                    # 处理与JSON格式相同的逻辑
                    existing_idx = self.danger_alerts.index[self.danger_alerts['物体'] == label].tolist()
                    if existing_idx:
                        for idx in existing_idx:
                            if self.danger_alerts.loc[idx, '状态'] == '等待LLM处理':
                                self.danger_alerts.loc[idx, '时间'] = time_str
                                self.danger_alerts.loc[idx, '置信度'] = float(confidence)
                                break
                        else:
                            if len(existing_idx) < 2:
                                self.danger_alerts = pd.concat([self.danger_alerts, new_alert], ignore_index=True)
                    else:
                        self.danger_alerts = pd.concat([self.danger_alerts, new_alert], ignore_index=True)
                    
                    if len(self.danger_alerts) > 50:
                        self.danger_alerts = self.danger_alerts.iloc[-50:]
            except Exception as e:
                print(f"解析危险警报失败: {str(e)}")
    
    def process_llm_response(self, message):
        """处理LLM响应消息（旧格式兼容）"""
        try:
            # 分隔前三个部分和响应内容
            first_part = message[:message.find(',', message.find(',', message.find(',') + 1) + 1) + 1]
            content_part = message[len(first_part):]
            
            parts = first_part.split(',')
            if len(parts) == 3:  # 消息头有3部分
                _, timestamp, label = parts
                response_text = content_part
                
                print(f"收到LLM响应，关于: {label}, 长度: {len(response_text)}")
                
                # 存储LLM响应
                with self.lock:
                    self.llm_responses[label] = response_text
                    
                    # 更新对应警报的状态
                    mask = self.danger_alerts['物体'] == label
                    self.danger_alerts.loc[mask, '状态'] = '已处理'
                
                # 通知UI可以更新了
                self.llm_response_updated.set()
            else:
                print(f"LLM响应格式不正确")
        except Exception as e:
            print(f"解析LLM响应失败: {str(e)}")
    
    def request_image(self):
        """请求图像数据"""
        try:
            # 创建临时连接
            image_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            image_socket.connect(('192.168.3.146', 5001))  # 连接到发送端图像服务器
            
            # 发送请求
            image_socket.sendall("REQUEST_IMAGE".encode('utf-8'))
            
            # 接收图像大小
            size_data = image_socket.recv(4)
            if not size_data:
                print("未收到图像大小数据")
                image_socket.close()
                return None
                
            size = struct.unpack("!I", size_data)[0]
            print(f"准备接收图像，大小: {size}字节")
            
            # 接收图像数据
            data = bytearray()
            while len(data) < size:
                chunk = image_socket.recv(min(size - len(data), 4096))
                if not chunk:
                    break
                data.extend(chunk)
            
            # 关闭连接
            image_socket.close()
            
            if len(data) != size:
                print(f"图像数据不完整: {len(data)}/{size}")
                return None
                
            # 解码图像
            img_array = np.frombuffer(data, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is not None:
                # 转BGR到RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
            else:
                print("无法解码图像")
                return None
                
        except Exception as e:
            print(f"获取图像错误: {str(e)}")
            return None

    def get_table(self):
        """获取数据表格"""
        with self.lock:
            return self.df.copy()
    
    def get_danger_alerts(self):
        """获取危险物品警报表格"""
        with self.lock:
            return self.danger_alerts.copy()
    
    def get_llm_response(self, label):
        """获取特定标签的LLM响应"""
        with self.lock:
            response = self.llm_responses.get(label)
            if response:
                return response
            return "正在生成处置方案..."

def create_ui():
    """创建Gradio用户界面"""
    receiver = Receiver()
    
    # 启动消息接收服务器线程
    server_thread = threading.Thread(target=receiver.start_server, args=(5000,))
    server_thread.daemon = True
    server_thread.start()
    
    # 使用基础主题
    with gr.Blocks(theme=gr.themes.Soft()) as ui:
        gr.Markdown("""
        # 基于端侧LLM的无人机动态环境决策响应系统
        ## 实时检测与智能决策平台
        """)
        
        # 创建用于自动刷新的状态变量
        refresh_state = gr.State(value=0)
        selected_alert = gr.State(value=None)
        
        # 创建标签页
        with gr.Tabs():
            # 标签页1: 常规检测
            with gr.TabItem("📊 实时检测"):
                with gr.Row():
                    # 左侧为表格
                    with gr.Column(scale=1):
                        regular_table = gr.Dataframe(
                            headers=['时间', '物体', '置信度'],
                            datatype=['str', 'str', 'number'],
                            label="🔍 检测结果",
                            interactive=False
                        )
                    
                    # 右侧为图像
                    with gr.Column(scale=1):
                        image_output = gr.Image(label="📷 当前检测图像")
                        gr.Button("🔄 刷新图像").click(
                            fn=receiver.request_image, 
                            inputs=[], 
                            outputs=[image_output]
                        )
            
            # 标签页2: 危险物品警报
            with gr.TabItem("⚠️ 危险物品警报"):
                with gr.Row():
                    # 左侧为警报表格
                    with gr.Column(scale=1):
                        alerts_table = gr.Dataframe(
                            headers=['时间', '物体', '置信度', '状态'],
                            datatype=['str', 'str', 'number', 'str'],
                            label="⚠️ 危险物品警报",
                            interactive=False,
                            elem_id="alerts_table"
                        )
                        
                        # 辅助函数，用于选择警报
                        def select_alert(evt: gr.SelectData):
                            # 当用户选择一行时返回选中的标签
                            row = evt.index[0]
                            if row < len(receiver.danger_alerts):
                                label = receiver.danger_alerts.iloc[row]['物体']
                                print(f"选择了警报: {label}")
                                return label
                            return None
                        
                        alerts_table.select(select_alert, inputs=[], outputs=[selected_alert])
                    
                    # 右侧为LLM响应
                    with gr.Column(scale=1):
                        llm_response_text = gr.HTML(
                            label="🤖 处置方案",
                            value="<div style='height:400px;overflow-y:auto;padding:10px;'>选择警报，查看LLM推理处置方案</div>"
                        )
                        
                        def update_response_display(label):
                            """获取并更新LLM响应显示"""
                            if not label:
                                return "<div style='height:400px;overflow-y:auto;padding:10px;'>请选择警报，查看处置方案</div>"
                            
                            # 获取响应
                            response = receiver.get_llm_response(label)
                            
                            # Markdown到HTML的转换
                            html_content = convert_markdown_to_html(response)
                            
                            # 包装在div中控制样式
                            return f"<div style='height:400px;overflow-y:auto;padding:10px;'>{html_content}</div>"
                        
                        def convert_markdown_to_html(text):
                            """将Markdown格式转换为HTML"""
                            if not text or text == "正在生成处置方案...":
                                return "<p>正在生成处置方案...</p>"
                                
                            # 1. 处理段落
                            text = text.replace("\n\n", "</p><p>")                            
                            # 2. 处理换行
                            text = text.replace("\n", "<br>")                          
                            # 3. 处理粗体 (**text**)
                            i = 0
                            result = ""
                            bold_open = False
                            while i < len(text):
                                if i+1 < len(text) and text[i:i+2] == "**":
                                    if bold_open:
                                        result += "</b>"
                                    else:
                                        result += "<b>"
                                    bold_open = not bold_open
                                    i += 2
                                else:
                                    result += text[i]
                                    i += 1
                            text = result
                            
                            # 4. 处理标题 (# text)
                            for i in range(6, 0, -1):
                                pattern = "<br>" + "#" * i + " "
                                replacement = f"<br><h{i}>"
                                text = text.replace(pattern, replacement)
                                # 添加标题结束标签
                                lines = text.split("<br>")
                                for j, line in enumerate(lines):
                                    if line.startswith(f"<h{i}>"):
                                        lines[j] = line + f"</h{i}>"
                                text = "<br>".join(lines)
                            
                            # 5. 处理列表项 (* text)
                            text = text.replace("<br>* ", "<br>• ")
                            text = text.replace("<br>- ", "<br>• ")
                            
                            # 6. 处理数字列表
                            lines = text.split("<br>")
                            for i, line in enumerate(lines):
                                if line.strip() and line.strip()[0].isdigit() and '. ' in line:
                                    # 将数字列表项变为加粗数字
                                    parts = line.split('. ', 1)
                                    if len(parts) > 1:
                                        lines[i] = f"<b>{parts[0]}.</b> {parts[1]}"
                            text = "<br>".join(lines)
                            
                            # 7. 包装成段落
                            if not text.startswith("<p>"):
                                text = "<p>" + text + "</p>"
                            
                            return text
                        
                        # 当选择警报时更新LLM响应
                        selected_alert.change(
                            fn=update_response_display, 
                            inputs=[selected_alert], 
                            outputs=[llm_response_text]
                        )
                        
                        # 添加手动刷新按钮
                        gr.Button("🔄 刷新处置方案").click(
                            fn=update_response_display, 
                            inputs=[selected_alert], 
                            outputs=[llm_response_text]
                        )
        
        # 添加状态信息区域
        with gr.Row():
            status_text = gr.Markdown("系统状态: 运行中")
        
        # 自动更新函数
        def update_all_tables(state):
            """更新所有表格并增加计数器以触发UI刷新"""
            return receiver.get_table(), receiver.get_danger_alerts(), state + 1
        
        # 自动刷新逻辑 - 只刷新检测结果表格
        ui.load(
            update_all_tables, 
            inputs=[refresh_state], 
            outputs=[regular_table, alerts_table, refresh_state], 
            every=1
        )
        
        # 处理UI更新的函数 - 修改为只在接收到LLM响应时更新一次
        def handle_ui_updates():
            """处理UI更新 - 只在收到新的LLM响应时更新一次"""
            while True:
                try:
                    # 等待LLM响应更新事件
                    receiver.llm_response_updated.wait()
                    receiver.llm_response_updated.clear()
                    
                    # 检查是否有最新响应标签
                    latest_label = receiver.latest_response_label
                    current_label = selected_alert.value
                    
                    # 如果有最新响应且当前未选择或选择的就是最新响应的标签
                    if latest_label and (not current_label or current_label == latest_label):
                        # 自动选择最新的警报
                        selected_alert.update(value=latest_label)
                        
                        # 直接更新UI组件
                        try:
                            response = receiver.get_llm_response(latest_label)
                            if response != "正在生成处置方案...":
                                # 转换为HTML
                                html_content = convert_markdown_to_html(response)
                                formatted_content = f"<div style='height:400px;overflow-y:auto;padding:10px;'>{html_content}</div>"
                                
                                # 更新文本区域
                                llm_response_text.update(value=formatted_content)
                                print(f"已自动更新UI显示: {latest_label}")
                        except Exception as e:
                            print(f"更新UI失败: {str(e)}")
                    
                    # 确保表格也更新一次
                    alerts_table.update(receiver.get_danger_alerts())
                    
                    # 防止CPU过度占用
                    time.sleep(0.5)
                except Exception as e:
                    print(f"UI更新处理错误: {str(e)}")
                    time.sleep(1)
        
        # 启动UI更新处理线程
        update_thread = threading.Thread(target=handle_ui_updates, daemon=True)
        update_thread.start()
    
    return ui

# 启动Web界面
if __name__ == "__main__":
    ui = create_ui()
    ui.launch(server_name="192.168.3.153", server_port=7860, share=True)
