from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import logging
import tensorflow as tf
import sys 
import spacy
from model import Model, _START_VOCAB
import time
import random
from corpus.match_triples import match_triples
import json
import numpy as np
random.seed(time.time())
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

# 新建一个基于查询模型的会话机器人ChatterBot作为替补，查询模型的机器人回复的语料比较安全但是效果一般，依赖语料
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

# 模型参数设置
tf.app.flags.DEFINE_boolean("is_train", False, "Set to True to train.")
tf.app.flags.DEFINE_integer("symbols", 30000, "vocabulary size.")
tf.app.flags.DEFINE_integer("embed_units", 300, "Size of word embedding.")
tf.app.flags.DEFINE_integer("trans_units", 100, "Size of trans embedding.")
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_entities", 21471, "entitiy vocabulary size.")
tf.app.flags.DEFINE_integer("num_relations", 44, "relation size.")
tf.app.flags.DEFINE_integer("layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")

FLAGS = tf.app.flags.FLAGS
if FLAGS.train_dir[-1] == '/':
    FLAGS.train_dir = FLAGS.train_dir[:-1]

f = open('%s/resource.txt' % FLAGS.data_dir)
resource_data = json.load(f)
f.close()

csk_triples = resource_data['csk_triples']
csk_entities = resource_data['csk_entities']
raw_vocab = resource_data['vocab_dict']
kb_dict = resource_data['dict_csk']

def gen_batched_data(data):
    encoder_len = max([len(item['post']) for item in data])+1
    decoder_len = max([len(item['response']) for item in data])+1
    triple_num = max([len(item['all_triples']) for item in data])+1
    triple_len = max([len(tri)
                      for item in data for tri in item['all_triples']])
    max_length = 20
    posts, responses, posts_length, responses_length = [], [], [], []
    entities, triples, matches, post_triples, response_triples = [], [], [], [], []
    match_entities, all_entities = [], []
    match_triples, all_triples = [], []
    # Not A Factor 用于对应停用词
    NAF = ['_NAF_H', '_NAF_R', '_NAF_T']

    # Padding补全句子长度
    def padding(sent, l):
        return sent + ['_EOS'] + ['_PAD'] * (l-len(sent)-1)

    def padding_triple(triple, num, l):
        newtriple = []
        triple = [[NAF]] + triple
        for tri in triple:
            newtriple.append(
                tri + [['_PAD_H', '_PAD_R', '_PAD_T']] * (l-len(tri)))
        pad_triple = [['_PAD_H', '_PAD_R', '_PAD_T']] * l
        return newtriple + [pad_triple] * (num - len(newtriple))

    for item in data:
        posts.append(padding(item['post'], encoder_len))
        responses.append(padding(item['response'], decoder_len))
        posts_length.append(len(item['post'])+1)
        responses_length.append(len(item['response'])+1)
        all_triples.append(padding_triple([[csk_triples[x].split(
            ', ') for x in triple] for triple in item['all_triples']], triple_num, triple_len))
        post_triples.append([[x] for x in item['post_triples']] +
                            [[0]] * (encoder_len - len(item['post_triples'])))
        response_triples.append([NAF] + [NAF if x == -1 else csk_triples[x].split(', ')
                                         for x in item['response_triples']] + [NAF] * (decoder_len - 1 - len(item['response_triples'])))
        match_index = []
        for idx, x in enumerate(item['match_index']):
            _index = [-1] * triple_num
            if x[0] == -1 and x[1] == -1:
                match_index.append(_index)
            else:
                _index[x[0]] = x[1]
                t = all_triples[-1][x[0]][x[1]]
                assert(t == response_triples[-1][idx+1])
                match_index.append(_index)
        match_triples.append(
            match_index + [[-1]*triple_num]*(decoder_len-len(match_index)))

        if not FLAGS.is_train:
            entity = [['_NONE']*triple_len]
            for ent in item['all_entities']:
                entity.append([csk_entities[x] for x in ent] +
                              ['_NONE'] * (triple_len-len(ent)))
            entities.append(entity+[['_NONE']*triple_len]
                            * (triple_num-len(entity)))

    batched_data = {'posts': np.array(posts),
                    'responses': np.array(responses),
                    'posts_length': posts_length,
                    'responses_length': responses_length,
                    'triples': np.array(all_triples),
                    'entities': np.array(entities),
                    'posts_triple': np.array(post_triples),
                    'responses_triple': np.array(response_triples),
                    'match_triples': np.array(match_triples)}

    return batched_data

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
            model = Model(
                FLAGS.symbols,
                FLAGS.embed_units,
                FLAGS.units,
                FLAGS.layers,
                embed = None,
                num_entities = FLAGS.num_entities + FLAGS.num_relations,
                num_trans_units = FLAGS.trans_units)
            # 从最新的的checkpoint加载训练好的模型参数
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
            print('restore from %s' % model_path)
            model.saver.restore(sess, model_path)
            # 首先尝试用系统模型回复
            # 词干化输入
            doc = nlp(userText)
            userText_post = [token.orth_.lower() for token in doc]
            print(userText_post)
            user_input = {}
            user_input["post"] = userText_post
            user_input["response"] = []
            data_input = match_triples(user_input, resource_data)
            batched_data = gen_batched_data([data_input])
            # 调用模型
            responses, ppx_loss = sess.run(['decoder_1/generation:0', 'decoder/ppx_loss:0'], {'enc_inps:0': batched_data['posts'], 'enc_lens:0': batched_data['posts_length'], 'dec_inps:0': batched_data['responses'], 'dec_lens:0': batched_data['responses_length'],
                                                                                              'entities:0': batched_data['entities'], 'triples:0': batched_data['triples'], 'match_triples:0': batched_data['match_triples'], 'enc_triples:0': batched_data['posts_triple'], 'dec_triples:0': batched_data['responses_triple']})
            result = []
            for token in responses[0]:
                if token != '_EOS':
                    result.append(token.decode())
                else:
                    break
            responseText = " ".join(result)           
    except BaseException as e:
        # 如果模型回复失败，使用Chatterbot的回复
        responseText = str(english_bot.get_response(userText))
    return responseText

if __name__ == "__main__":
    app.run()
