import json
import spacy

# 初始化spaCy nlp
nlp = spacy.load('en_core_web_sm')

'''方法用于根据id获取message
'''
def get_message_by_id(messages, references, id):
    for message in messages:
        if id == message['id']:
            return message
    for reference in references:
        if id == reference['id'] and 'message_type' in reference and 'update' == reference['message_type']:
            return reference

def clean_text(text):
    doc = nlp(text)
    lemma = [token.orth_.lower() for token in doc]
    return lemma

conversations = []

# 整理原yammer语料文件，从中找出post和response
with open('/Users/charles/dev/digitalsuit-bot/corpus/raw.json') as f:
    data = json.load(f)
    messages = data['messages']
    references = data['references']
    for message in messages:
        # print(message['id'])
        conversation = {}
        if message['replied_to_id']:
            post_message = get_message_by_id(messages, references, message['replied_to_id'])
            
            if post_message and 'body' in post_message:
                if 'plain' in post_message['body']:
                    conversation['post'] = clean_text(post_message['body']['plain'])
                elif 'parsed' in post_message['body']:
                    conversation['post'] = clean_text(post_message['body']['parsed'])
            
            if 'body' in message:
                if 'plain' in message['body']:    
                    conversation['response'] = clean_text(message['body']['plain'])
                elif 'parsed' in message['body']:
                    conversation['response'] = clean_text(message['body']['parsed'])
            
            conversations.append(conversation)

print(conversations)

with open('/Users/charles/dev/digitalsuit-bot/corpus/test.text', 'w') as f:
    for conversation in conversations:
        jsonStr = json.dumps(conversation)
        f.write(jsonStr)
        f.write('\n')

