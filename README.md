# ds-bot 融合了知识图谱的会话机器人
## digitalsuite-bot 

### 主要目录结构
- chatbot_app 目录包含用ChatterBot实现的web版会话机器人外壳，安装依赖 `pip install -r requirements.txt`，执行 `python app.py` 可运行, 运行在 `http://localhost:5000`
- conceptnet_demo 目录包含抽取conceptnet的方法和绘制图谱用于展示的html文件（使用d3.js）
- corpus 目录包含从yammer抓取原始语料的样例（使用yampy SDK），以及从原始语料中整理出对话
- data 目录用于程序读取训练相关的语料和数据，数据体积比较大，需要另外下载所有数据文件并解压拷贝到此目录，下载链接见数据链接小节
- seq2seq-old 目录包含老的seq2seq模型
- tools 包含工具类
- model.py 为模型主文件
- main.py 为训练测试入口，读取所有训练语料和数据并执行模型训练或者测试

### 数据链接
- ConceptNet 5.7.0 https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
- GloVe http://nlp.stanford.edu/data/glove.840B.300d.zip
- 开发测试语料（非生产） 

### 开发环境版本
- Python 3.6.2 64-bit
- Tensorflow 1.13.2
