# import warnings
# warnings.filterwarnings("ignore")
import spacy
import neuralcoref
nlp = spacy.load('en_core_web_sm')

neuralcoref.add_to_pipe(nlp)

QA = {"Who is Abraham Lincoln?":"An American statesman and lawyer who served as the 16th President of the United States.",
      "When was Abraham Lincoln born?":"February 12, 1809.",
      "Where is Abraham Lincoln's hometown?":"Hodgenville, Kentucky"}
para = "The woman reading a newspaper, she sat on the bench with her dog."
doc = nlp(para)
print(doc._.has_coref)
print(doc._.coref_clusters)
print(doc._.coref_resolved)