# import warnings
# warnings.filterwarnings("ignore")
import spacy
import neuralcoref
nlp = spacy.load('en_core_web_sm')

# 指代消解器
neuralcoref.add_to_pipe(nlp)

QA = {"Who is Abraham Lincoln?":"An American statesman and lawyer who served as the 16th President of the United States.",
      "When was Abraham Lincoln born?":"February 12, 1809.",
      "Where is Abraham Lincoln's hometown?":"Hodgenville, Kentucky"}
para = "The woman reading a newspaper, she sat on the bench with her dog."
doc = nlp(para)
print(doc._.has_coref)
print(doc._.coref_clusters)
print(doc._.coref_resolved)
# 分词
token = [t for t in doc]
# 分词 orth_ 可以识别标点符号
token2 = [token.orth_ for token in doc]
# 词干化
lemma = [l.lemma_ for l in doc]
# 词性标注
# pos = [p.pos_ for p in doc]