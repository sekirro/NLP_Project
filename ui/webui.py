import streamlit as st
import pandas as pd
import docx
import pdfplumber
import requests
import json
import tkinter as tk
from tkinter import filedialog

def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def get_file_list():
    """获取OSS中的文件列表"""
    try:
        response = requests.get('http://localhost:5000/files')
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                return data['files']
        return []
    except Exception as e:
        st.error(f"获取文件列表失败: {str(e)}")
        return []

def delete_file(filename):
    """从OSS删除文件"""
    try:
        response = requests.delete(f'http://localhost:5000/delete/{filename}')
        if response.status_code == 200:
            return True
        return False
    except Exception as e:
        st.error(f"删除文件失败: {str(e)}")
        return False

def download_file(filename):
    """从OSS下载文件"""
    try:
        # 强制将tkinter窗口置顶
        root = tk.Tk()
        root.wm_attributes('-topmost', 1)  # 设置窗口置顶
        root.withdraw()  # 隐藏主窗口
        
        # 确保对话框显示在最前面
        save_path = filedialog.askdirectory(
            title=f"选择保存 {filename} 的位置",
            parent=root  # 设置父窗口
        )
        
        root.destroy()
        
        if not save_path:  # 用户取消选择
            return False
            
        response = requests.post(
            f'http://localhost:5000/download/{filename}',
            json={'save_path': save_path},
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            return True
        return False
    except Exception as e:
        st.error(f"下载文件失败: {str(e)}")
        return False

def update_knowledge_base():
    """更新知识库"""
    try:
        response = requests.post(
            'http://localhost:5000/update',
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 200:
            return True
        return False
    except Exception as e:
        st.error(f"更新知识库失败: {str(e)}")
        return False

def send_question(prompt, model_type="without RAG"):
    """发送问题到后端服务器"""
    try:
        response = requests.post(
            'http://localhost:5000/qa',  # 本地后端服务器地址
            json={'question': prompt,
                  'model_type': model_type},
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                return data['answer']
        return "抱歉，服务器出现错误。"
    except Exception as e:
        return f"连接服务器失败: {str(e)}"

def upload_file_to_server(file):
    """上传文件到后端服务器"""
    try:
        files = {'file': file}
        response = requests.post(
            f'http://localhost:5000/upload/{file.name}',
            files=files
        )
        if response.status_code == 200:
            return True
        return False
    except Exception as e:
        st.error(f"文件上传失败: {str(e)}")
        return False

def main():
    st.set_page_config(
        page_title="老中医AI",
        page_icon=":robot_face:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 修改 CSS 样式
    st.markdown(
        """
        <style>
        .main > div {
            padding-bottom: 100px;
        }
        .stForm {
            position: fixed;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 50%;
            max-width: 800px;
            background-color: white;
            padding: 1rem;
            border-top: 1px solid #ddd;
            z-index: 100;
            margin-left: calc(15.625rem / 2);
        }
        .stSidebar {
            z-index: 200;
        }
        .file-list {
            margin-top: 20px;
        }
        .stButton > button[kind="primary"] {
            background-color: #0099FF;  /* 亮蓝色 */
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
    
        .stButton > button[kind="primary"]:hover {
            background-color: #1E88E5;  /* 深蓝色 */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("老中医AI")
    init_chat_history()

    # 侧边栏：文件上传组件和文件列表
    with st.sidebar:
        st.markdown("### 文件上传")
        uploaded_file = st.file_uploader("上传文件进行分析",
                                        type=['txt','docx','pdf','xlsx','csv',
                                            'cpp', 'py', 'c', 'h', 'hpp', 'java', 'js']
                                        )
        
        # 添加一个上传按钮
        col1, col2 = st.columns(2)
        with col1:
            upload_button = st.button("⬆️ 上传文件", 
                                    type="primary",
                                    disabled=uploaded_file is None)
        with col2:
            update_button = st.button("🔄 更新知识库", 
                                    type="primary",
                                    use_container_width=True)
        
        # 处理文件上传
        if uploaded_file is not None and upload_button:
            if upload_file_to_server(uploaded_file):
                st.session_state['show_success'] = f"文件 {uploaded_file.name} 上传成功！"
                st.session_state.messages = []
        
        # 处理知识库更新
        if update_button:
            with st.spinner("⏳ 正在更新知识库..."):
                if update_knowledge_base():
                    st.success("✅ 知识库更新成功！")
                    st.rerun()
                else:
                    st.error("❌ 知识库更新失败")
        
        # 显示文件列表
        st.markdown("### 文件列表")
        files = get_file_list()
        
        if not files:
            st.info("暂无文件")
        else:
            for file in files:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(file['name'])
                with col2:
                    if st.button("下载", key=f"download_{file['name']}"):
                        if download_file(file['name']):
                            st.session_state['show_success'] = f"文件 {file['name']} 下载成功！"
                        else:
                            st.session_state['show_error'] = "下载失败"
                with col3:
                    if st.button("删除", key=f"delete_{file['name']}"):
                        if delete_file(file['name']):
                            st.session_state['show_success'] = f"文件 {file['name']} 删除成功！"
                            st.rerun()
                        else:
                            st.session_state['show_error'] = "删除失败"
        
        # 在文件列表下方显示成功/错误消息
        if 'show_success' in st.session_state:
            st.success(st.session_state['show_success'])
            del st.session_state['show_success']
        if 'show_error' in st.session_state:
            st.error(st.session_state['show_error'])
            del st.session_state['show_error']

    # 创建主要内容区
    main_content = st.container()
    
    # 显示聊天历史
    with main_content:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
                st.markdown(message["content"])

    # 用户输入区域
    with st.form(key="chat_form", clear_on_submit=True):
        cols = st.columns([2, 6, 1])  # 调整列宽以适应复选框
        with cols[0]:
            model_type = st.selectbox("选择模型", 
                                      ["without RAG", "RAG combined"], 
                                      label_visibility="collapsed",
                                      )
        with cols[1]:
            prompt = st.text_input("请输入您的问题", key="user_input", label_visibility="collapsed")
        with cols[2]:
            submit_button = st.form_submit_button("发送", use_container_width=True)
        
        if submit_button and prompt:
            st.session_state.messages.append({"role": "user", "content": prompt, "model_type": model_type})
            st.rerun()

    # 检查是否需要获取AI回复
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.spinner("AI正在思考中..."):
            last_message = st.session_state.messages[-1]
            response = send_question(
                last_message["content"], 
                model_type=last_message["model_type"]
            )
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

if __name__ == "__main__":
    main()