## 要求的文件

1. PPT和技术文档在根文件夹
2. 评估代码在evaluate/evaluate_code
3. 评估结果和绘制代码在evaluate/evaluate_result
4. 微调代码在finetune文件夹
5. 系统实现代码在ui文件夹

## 项目如何启动

#### 前端

1. 本地运行ui文件夹下的local_backend.py文件
2. 本地命令行切换到ui文件夹

```cmd
cd ui
```

3. 本地输入命令启动前端

```cmd
streamlit run webui.py
```

#### 服务器后端（获取大模型回复）

1. 在[ngrok](https://ngrok.com/)官网获取authtoken，填入ui文件夹back_stream.ipynb对应的位置
2. 在[阿里云](https://www.aliyun.com)获取ACCESSKEY_ID和ACCESSKEY_SECRET，填入ui文件夹back_stream.ipynb相应位置
3. 启动阿里云的oss服务，bucket命名为csgroup（也可以改其他的）
4. 将ui文件夹中的back_streamlit.ipynb文件大模型上传至服务器
5. 将大模型上传至服务器
6. 修改back_streamlit.ipynb文件中的路径（改成服务器中的对应路径）（linux下`pwd`查看当前路径）
7. 运行back_streamlit.ipynb

### 微调大模型

1. 将原始模型上传至服务器
2. 将数据集上传至服务器（需要是datasets格式，可以用data_transform.ipynb转换可以将dataset_transformed文件夹上传）可以将dataset_transformed文件夹上传，已经转换好
3. 修改finetune中的lora_sft_qwen2.5-7B.ipynb文件的对应路径
4. 运行finetune中的lora_sft_qwen2.5-7B.ipynb文件（数据集处理部分可以根据不同格式更改）