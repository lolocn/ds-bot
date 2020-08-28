# 融合了知识库的会话机器人
## digitalsuite-bot 智能会话模块

### 主要目录结构
- model.py 为模型主文件
- main.py 为训练测试入口，读取所有训练语料和数据并执行模型训练或者测试
- app.py 为用ChatterBot实现的web版会话机器人外壳。
安装依赖 `pip install -r requirements.txt`。如果有训练好的参数，执行 `python app.py` 可运行, 访问 `http://localhost:5000`可以使用会话机器人
- conceptnet_demo 目录包含抽取conceptnet的方法和绘制图谱用于展示的html文件（使用d3.js）
- corpus 目录包含从yammer抓取原始语料的样例yammer.py（使用yampy SDK），从原始语料中整理出对话coversation.py，用于匹配对话和三元组的match_triples.py
- data 目录用于程序读取训练相关的语料和数据，数据体积比较大，需要另外下载所有数据文件并解压拷贝到此目录，下载链接见数据链接小节
- train 目录用于存放模型训练的checkpoint记录
- seq2seq-old 目录包含老的seq2seq模型
- tools 目录包含工具类
- templates 为web前端UI
- static 为web前端的资源文件css


### 数据链接
- ConceptNet 5.7.0 https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
- GloVe http://nlp.stanford.edu/data/glove.840B.300d.zip
- 开发测试语料样例以及21000步训练checkpoint 链接: https://pan.baidu.com/s/1SQ5GFIdut4Q5UeWWQNAhXQ  提取码: e8bu

### 开发环境及工具版本
- Python 3.6.2 64-bit
- Tensorflow 1.13.2
- spaCy 2.1.8
