import configparser
from datetime import datetime
from pymongo import MongoClient
import itertools
import spacy

nlp = spacy.load('en')

### init and read config
config = configparser.ConfigParser()
config.read('./config.ini')

### connect to mongodb
MongoDB = config["MLDoc"]["Database"]
MongoUser = config["MLDoc"]["User"]
MongoPW = config["MLDoc"]["PW"]

uri = "mongodb://" + MongoUser + ":" + MongoPW + "@140.117.69.70:30241/" + MongoDB + "?authMechanism=SCRAM-SHA-1"

client = MongoClient(uri)
db = client.MLDoc

# update function is designed to help pymongo update nlp results into Database

def update(target_collection, doc_id, sentences, nested_token_list):

    target_collection.update_one({"_id": doc_id},
                      {
                          "$set":{
                          "sentences": sentences,
                          "nested_token_list": nested_token_list,
                          "nlp_process": True
                          }
                      })

def error_update(target_collection, doc_id):

    target_collection.update_one({"_id": doc_id},
                      {
                          "$set":{
                          "nlp_error": True
                      }
                    })

target_collection = db.English

docs = target_collection.find({"nlp_process":{"$exists": False}},{"_id":1, "Content":1}, no_cursor_timeout=True)

gen1, gen2 = itertools.tee(docs)
ids = (doc["_id"] for doc in gen1 if len(doc["Content"]) >0)
texts = (doc["Content"] for doc in gen2 if len(doc["Content"]) >0)

# improve version
start_time = datetime.now()

error_list = list()
index = 0

docus = nlp.pipe(texts,batch_size=50,n_threads = -1)

# for each_document in nlp.pipe(texts, batch_size=20, n_threads=-1):
for id_, each_document in zip(ids, docus):
#     sentences = [sentence.text for sentence in each_document.sents]
    nested_token_list = []
    sentences= []
    for sentence in each_document.sents:
        token_list = []
        sentences.append(sentence.text)
        for token in sentence:
            each_token = {}
            if token.is_space != True:
                each_token["text"] = token.text
                each_token["lemma_"] = token.lemma_
                each_token["pos_"] = token.pos_
                each_token["shape_"] = token.shape_
                each_token["is_alpha"] = token.is_alpha
                each_token["is_stop"] = token.is_stop
                token_list.append(each_token)
        nested_token_list.append(token_list)


    try:
         update(target_collection, id_, sentences, nested_token_list)
    except:
         error_list.append(id_)


    index += 1
    if(index % 1000 ==0):
        time_elapsed = datetime.now() - start_time
        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

print(error_list)
docs.close()

time_elapsed = datetime.now() - start_time

print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
