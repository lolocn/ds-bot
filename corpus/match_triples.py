import json

'''
方法用于匹配post和response中匹配到的triple
样例：
{
    "post": [],
    "response": [],
    "post_triples": [0, 0, 0, 1, 0, 0, 2, 0], #标注post中的词匹配到的entity，0表示没有找到，大于0表示找到entity
    "all_triples": [],  #列出post和response所有匹配到的triples
    "all_entities": [], #列出post和response所有匹配到的entity
    "response_triples": [], #response匹配的triple，-1表示没有match
    "match_index": [[-1, -1], [-1, -1], [1, 23]] #表示response中的entity和post匹配，长度为response的长度，每组第一个数字表示post中和当前单词match的index，第二个数字表示match到的all_entities的index， [-1,-1]表示没有匹配
    "match_triples": [] #表示post和response正好匹配到的triples的index
}

'''
def match_triples(test, data):

    data['postEntityToCSKTripleIndex'] = {}
    data['postEntityToOtherCSKTripleEntities'] = {}
    index = 0
    for triple in data['csk_triples']:
        firstEntity = triple.split(',')[0]
        secondEntity = triple.split(',')[2].strip()
        if(not firstEntity in data['postEntityToCSKTripleIndex']):
            data['postEntityToCSKTripleIndex'][firstEntity] = []
        data['postEntityToCSKTripleIndex'][firstEntity].append(index)
        if(not secondEntity in data['postEntityToCSKTripleIndex']):
            data['postEntityToCSKTripleIndex'][secondEntity] = []
        data['postEntityToCSKTripleIndex'][secondEntity].append(index)

        if (not firstEntity in data['postEntityToOtherCSKTripleEntities']):
            data['postEntityToOtherCSKTripleEntities'][firstEntity] = []
        data['postEntityToOtherCSKTripleEntities'][firstEntity].append(data['dict_csk_entities'][secondEntity])
        if (not secondEntity in data['postEntityToOtherCSKTripleEntities']):
            data['postEntityToOtherCSKTripleEntities'][secondEntity] = []
        data['postEntityToOtherCSKTripleEntities'][secondEntity].append(data['dict_csk_entities'][firstEntity])
        index += 1

    data['indexToCSKTriple'] = {v: k for k,v in data['dict_csk_triples'].items()}

    post_triples = []
    all_triples = []
    all_entities = []

    post = test['post']
    index = 0
    for word in post:
        try:
            entityIndex = data['dict_csk_entities'][word]
            index += 1
            post_triples.append(index)
            all_triples.append(data['postEntityToCSKTripleIndex'][word])
            all_entities.append(data['postEntityToOtherCSKTripleEntities'][word])
        except:
            post_triples.append(0)
    test['post_triples'] = post_triples
    test['all_triples'] = all_triples
    test['all_entities'] = all_entities

    response_triples = []
    match_index = []
    match_triples = []
    for word in test['response']:
        try:
            found = False
            entityIndex = data['dict_csk_entities'][word]
            for index,entitiesList in enumerate(test['all_entities']):
                for subindex,entity in enumerate(entitiesList):
                    if(entity == entityIndex):
                        match_index.append([index+1,subindex])
                        response_triples.append(test['all_triples'][index][subindex])
                        match_triples.append(test['all_triples'][index][subindex])
                        found = True
                        break
            if not found:
                response_triples.append(-1)
                match_index.append([-1,-1])
        except:
            response_triples.append(-1)
            match_index.append([-1,-1])

    test['response_triples'] = response_triples
    test['match_index'] = match_index
    test['match_triples'] = match_triples
    print(str(test))
    return test

# 开发测试的test，可以看到效果
# f = open('../data/resource.txt')
# data = json.load(f)
# f.close()
# test = {"post": ["you", "mean", "the", "occupation", "that", "did", "happen", "?"], "response": ["no", "i", "mean", "the", "fighting", "invasion", "that", "the", "military", "made", "so", "many", "purple", "hearts", "for", "in", "anticipation", "for", "that", "we", "have", "n't", "used", "up", "to", "this", "day", "."]}
# match_triples(test, data)