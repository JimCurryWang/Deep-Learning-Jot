'''
Spacy: 
    Industrial-strength Natural Language Processing (NLP) in Python
    It features state-of-the-art speed and neural network models for tagging, parsing, 
    named entity recognition, text classification and more, 
    multi-task learning with pretrained transformers like BERT, 
    as well as a production-ready training system and easy model packaging, 
    deployment and workflow management

    + https://github.com/explosion/spaCy
    + https://spacy.io/models

$ pip install -U spacy 
$ python -m spacy download en_core_web_sm  # english
$ python -m spacy download zh_core_web_sm  # Chinese
$ python -m spacy download nl_core_news_sm # Dutch
$ python -m spacy download de_core_news_sm # German
'''

import spacy

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")
doc = nlp(text)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)

