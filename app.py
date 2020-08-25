from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import logging
import tensorflow as tf
import sys 
import spacy
from model import Model, _START_VOCAB
# from chatterbot.trainers import UbuntuCorpusTrainer

# 使用Flask框架构建app
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# 初始化spaCy nlp
nlp = spacy.load('en_core_web_sm')

# english_bot = ChatBot("Chatterbot", 
#     storage_adapter="chatterbot.storage.MongoDatabaseAdapter",
#     logic_adapters=[
#         'chatterbot.logic.BestMatch',
#         'chatterbot.logic.MathematicalEvaluation'
#     ],
#     filters=[
#         'chatterbot.filters.RepetitiveResponseFilter'
#     ],
#     database_uri="mongodb://127.0.0.1:27017/chatterbot-1w"
# )

# 新建一个使用查询模型的会话机器人ChatterBot作为替补，查询模型的机器人回复的语料比较安全但是效果不一定好
english_bot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
trainer = ChatterBotCorpusTrainer(english_bot)
trainer.train("chatterbot.corpus.english")

# ubuntu_bot = ChatBot("UbuntuBot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
# trainer = UbuntuCorpusTrainer(ubuntu_bot)
# trainer.train()

# 映射根请求返回会话前端web UI
@app.route("/")
def home():
    return render_template("index.html")

# 加载模型
tf.app.flags.DEFINE_integer("symbols", 30000, "vocabulary size.")
tf.app.flags.DEFINE_integer("embed_units", 300, "Size of word embedding.")
tf.app.flags.DEFINE_integer("trans_units", 100, "Size of trans embedding.")
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_entities", 21471, "entitiy vocabulary size.")
tf.app.flags.DEFINE_integer("num_relations", 44, "relation size.")
tf.app.flags.DEFINE_integer("layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")

FLAGS = tf.app.flags.FLAGS
if FLAGS.train_dir[-1] == '/':
    FLAGS.train_dir = FLAGS.train_dir[:-1]

model = Model(
            FLAGS.symbols,
            FLAGS.embed_units,
            FLAGS.units,
            FLAGS.layers,
            embed=None,
            num_entities = FLAGS.num_entities + FLAGS.num_relations,
            num_trans_units = FLAGS.trans_units)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# 映射API请求 获取回复
@app.route("/get")
def get_bot_response():
    responseText = 'hi'
    # 取得用户输入
    userText = request.args.get('msg')
    try:
        with tf.Session(config=config) as sess:
            # 从最新的的checkpoint加载训练好的模型参数
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
            print('restore from %s' % model_path)
            model.saver.restore(sess, model_path)
            # 首先尝试用系统模型回复
            # 词干化
            doc = nlp(userText)
            userText_post = [l.lemma_ for l in doc]
            print(userText_post)
            # 调用模型
            responses, ppx_loss = sess.run(['decoder_1/generation:0', 'decoder/ppx_loss:0'], {'enc_inps:0': userText_post, 'enc_lens:0': len(userText_post)})
            result = []
            for token in responses[0]:
                if token != '_EOS':
                    result.append(token.decode())
                else:
                    break
            responseText = " ".join(result)           
    except RuntimeError as e:
        # 项目的tricky方法，如果模型回复失败，使用Chatterbot的回复
        responseText = str(english_bot.get_response(userText))
    return responseText

if __name__ == "__main__":
    app.run()
