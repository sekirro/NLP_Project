from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
import oss2
import os
import requests
import time
from threading import Thread
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry  # 从 urllib3 导入 Retry

def init_oss():
    # OSS初始化配置
    accessKeyId = os.getenv('ACCESSKEY_ID')
    accessKeySecret = os.getenv('ACCESSKEY_SECRET')
    auth = oss2.Auth(accessKeyId, accessKeySecret)
    endpoint = 'http://oss-cn-beijing.aliyuncs.com'
    bucketName = 'csgroup'
    return oss2.Bucket(auth, endpoint, bucketName)
def get_colab_url():
    try:
        # 从OSS读取URL
        url_content = bucket.get_object('colab_url.txt').read()
        return url_content.decode('utf-8').strip()
    except Exception as e:
        print(f"获取Colab URL出错: {str(e)}")
        return None

def check_colab_connection():
    """检查Colab连接是否有效"""
    try:
        url = get_colab_url()
        if not url:
            return False
            
        # 尝试访问健康检查端点
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def update_colab_url():
    """定期检查并更新Colab URL"""
    while True:
        try:
            if not check_colab_connection():
                print("Colab连接断开，等待新URL...")
                # 等待新URL出现
                while True:
                    url = get_colab_url()
                    if url and check_colab_connection():
                        print(f"获取到新的Colab URL: {url}")
                        break
                    time.sleep(5)  # 每5秒检查一次
            time.sleep(30)  # 每30秒检查一次连接状态
        except Exception as e:
            print(f"更新URL时出错: {str(e)}")
            time.sleep(5)

def create_retry_session():
    """创建带有重试机制的会话"""
    session = requests.Session()
    retries = Retry(
        total=3,  # 总重试次数
        backoff_factor=0.5,  # 重试间隔
        status_forcelist=[500, 502, 503, 504],  # 需要重试的HTTP状态码
        allowed_methods=["GET", "POST"]  # 允许重试的请求方法
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

app = Flask(__name__)
CORS(app)
bucket = init_oss()

@app.route('/upload/<path:filename>', methods=['POST'])
def upload_file(filename):
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file part'
            }), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No selected file'
            }), 400

        bucket = init_oss()
        # 直接从请求中读取文件内容上传到OSS
        bucket.put_object(filename, file.read())
        
        return jsonify({
            'status': 'success',
            'message': f'File {filename} uploaded successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500



@app.route('/files', methods=['GET'])
def list_files():
    try:
        bucket = init_oss()
        files_list = []
        print("开始获取OSS文件列表")
        
        # 修正：使用 .object_iter 来遍历文件
        for obj in oss2.ObjectIterator(bucket):
            print(f"找到文件: {obj.key}")
            file_info = {
                'name': obj.key,
                'size': obj.size,
                'last_modified': obj.last_modified
            }
            files_list.append(file_info)
        
        print(f"总共找到 {len(files_list)} 个文件")
        return jsonify({
            'status': 'success',
            'files': files_list
        })
    except Exception as e:
        print(f"获取文件列表出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/download/<path:filename>', methods=['POST'])
def download_file(filename):
    try:
        data = request.get_json()
        if not data or 'save_path' not in data:
            return jsonify({
                'status': 'error',
                'message': '未指定保存路径'
            }), 400
            
        save_path = data['save_path']
        bucket = init_oss()
        
        # 构建完整的保存路径
        local_file = os.path.join(save_path, filename)
        
        # 从OSS下载文件
        bucket.get_object_to_file(filename, local_file)
        
        return jsonify({
            'status': 'success',
            'message': f'File {filename} downloaded successfully',
            'local_path': local_file
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# 在Flask后端添加删除文件的路由
@app.route('/delete/<path:filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        bucket = init_oss()
        # 从OSS删除文件
        bucket.delete_object(filename)
        
        return jsonify({
            'status': 'success',
            'message': f'File {filename} deleted successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/update', methods=['POST'])
def update_knowledge_base():
    """更新知识库"""
    try:
        # 获取Colab URL并发送更新请求
        url = get_colab_url()
        if not url:
            return jsonify({
                'status': 'error',
                'message': '无法获取Colab服务器地址'
            }), 500
            
        # 使用带重试机制的会话
        session = create_retry_session()
        update_url = url + "/update"
        
        # 发送更新请求到Colab服务器
        response = session.post(update_url)
        if response.status_code != 200:
            return jsonify({
                'status': 'error',
                'message': '更新知识库失败'
            }), 500
        
        return jsonify({
            'status': 'success',
            'message': '知识库更新成功'
        })
        
    except Exception as e:
        print(f"更新知识库时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/qa', methods=['POST'])
def question_answer():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'status': 'error',
                'message': '请提供问题'
            }), 400
            
        question = data['question']
        model_type = data.get('model_type', 'without RAG')  # 获取模型类型，默认为 RAG combined
        
        # 获取最新的Colab URL
        url = get_colab_url()
        if not url:
            return jsonify({
                'status': 'error',
                'message': '无法获取Colab服务器地址'
            }), 500
            
        # 使用带重试机制的会话
        session = create_retry_session()
        url = url + "/qa"
        print(f"使用模型: {model_type}")
        print(f"请求URL: {url}")

        # 发送请求时包含模型类型
        response = session.post(url, json={
            'question': question,
            'model_type': model_type
        })
        
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({
                'status': 'error',
                'message': '调用Colab服务失败'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)