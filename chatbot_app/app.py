from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import logging
import tensorflow as tf
import sys 
sys.path.append("..") 
from model import Model, _START_VOCAB
# from chatterbot.trainers import UbuntuCorpusTrainer

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

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

# english_bot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
# trainer = ChatterBotCorpusTrainer(english_bot)
# trainer.train("chatterbot.corpus.english")

# ubuntu_bot = ChatBot("UbuntuBot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
# trainer = UbuntuCorpusTrainer(ubuntu_bot)
# trainer.train()

# 根目录返回会话前端UI
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
tf.app.flags.DEFINE_string("train_dir", "../train", "Training directory.")

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
with tf.Session(config=config) as sess:
    # 加载最新的训练好的模型
    model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
    print('restore from %s' % model_path)
    model.saver.restore(sess, model_path)
    saver = model.saver

    # API请求 获取回复
    @app.route("/get")
    def get_bot_response():
        userText = request.args.get('msg')
        # responseText = str(english_bot.get_response(userText))
        return "test"


if __name__ == "__main__":
    app.run()
