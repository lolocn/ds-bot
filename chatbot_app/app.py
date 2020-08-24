from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from ../model import Model, _START_VOCAB
import logging
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
english_bot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
trainer = ChatterBotCorpusTrainer(english_bot)
trainer.train("chatterbot.corpus.english")

# ubuntu_bot = ChatBot("UbuntuBot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
# trainer = UbuntuCorpusTrainer(ubuntu_bot)
# trainer.train()


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    responseText = str(english_bot.get_response(userText))
    return responseText


if __name__ == "__main__":
    app.run()
