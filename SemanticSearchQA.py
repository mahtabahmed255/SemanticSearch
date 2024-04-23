import random
import os
import pickle
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

with open("//QAEmbeddings.pickle", "rb") as f:
    finalEmbeddings, data = pickle.load(f)

finalEmbeddings_ = []
flatData = dict()

with open("//whquestions.pickle", "rb") as f:
    _, wh_questions, _ = pickle.load(f)

# This example shows how to:
#   1. connect to Milvus server
#   2. create a collection
#   3. insert entities
#   4. create index
#   5. search


_HOST = '127.0.0.1'
_PORT = '19530'

# Const names
_COLLECTION_NAME_QUESTION = 'questionCollection'
_ID_FIELD_NAME_QUESTION = 'questionID'
_VECTOR_FIELD_NAME_QUESTION = 'questionVector'

_COLLECTION_NAME_ANSWER = 'answerCollection'
_ID_FIELD_NAME_ANSWER = 'answerID'
_VECTOR_FIELD_NAME_ANSWER = 'answerVector'

# Vector parameters
_DIM = 1024
_INDEX_FILE_SIZE = 32  # max file size of stored index

# Index parameters
_METRIC_TYPE = 'L2'
_INDEX_TYPE = 'HNSW' #'IVF_FLAT'
M = 1024
efConstruction = 200
efSearch = 200

_NLIST = 1024
_NPROBE = 16
_TOPKQ = 4
_TOPKA = 5


# Create a Milvus connection
def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=_HOST, port=_PORT)
    print(f"\nList connections:")
    print(connections.list_connections())


# Create a collection named 'demo'
def create_collection(name, id_field, vector_field):
    field1 = FieldSchema(name=id_field, dtype=DataType.INT64, description="int64", is_primary=True)
    field2 = FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, description="float vector", dim=_DIM,
                         is_primary=False)
    schema = CollectionSchema(fields=[field1, field2], description="collection description")
    collection = Collection(name=name, data=None, schema=schema, properties={"collection.ttl.seconds": 15})
    print("\ncollection created:", name)
    return collection

def has_collection(name):
    return utility.has_collection(name)


# Drop a collection in Milvus
def drop_collection(name):
    collection = Collection(name)
    collection.drop()
    print("\nDrop collection: {}".format(name))


# List all collections in Milvus
def list_collections():
    print("\nlist collections:")
    print(utility.list_collections())


def insert(collection, num, dim):
    if collection.name == _COLLECTION_NAME_QUESTION:
        Data = [
            [i for i in range(len(finalEmbeddings))],
            [j[0] for j in finalEmbeddings],
        ]
    else:
        ids = []
        embeddings = []
        k = 0
        for i in range(len(finalEmbeddings)):
            temp = []
            for j in range(len(finalEmbeddings[i]) - 1):
                ids.append(k)
                embeddings.append(finalEmbeddings[i][j+1])
                temp.append((k, finalEmbeddings[i][j+1]))
                flatData[k] = data[i][j+1]
                k += 1
            finalEmbeddings_.append(temp)
        Data = [
            ids,
            embeddings,
        ]
    collection.insert(Data)
    return Data[1]



def get_entity_num(collection):
    print("\nThe number of entity: ", collection.name)
    print(collection.num_entities)


def create_index(collection, filed_name):
    index_param = {
        "index_type": _INDEX_TYPE,
        #"params": {"nlist": _NLIST},
        "params": {"M": M, "efConstruction": efConstruction, "efSearch": efSearch},
        "metric_type": _METRIC_TYPE}
    collection.create_index(filed_name, index_param)
    print("\nCreated index:\n{}".format(collection.index().params))


def drop_index(collection):
    collection.drop_index()
    print("\nDrop index sucessfully")


def load_collection(collection):
    collection.load()


def release_collection(collection):
    collection.release()


def search(collection, vector_field, id_field, search_vectors, id):
    if not isinstance(search_vectors[0], list):
        search_vectors = [search_vectors]
    if collection.name == _COLLECTION_NAME_QUESTION:
        search_param = {
            "data": search_vectors,
            "anns_field": vector_field,
            "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
            "limit": _TOPKQ,
            "expr": "questionID >= 0"}
    else:
        search_param = {
            "data": search_vectors,
            "anns_field": vector_field,
            "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
            "limit": _TOPKA,
            "expr": "answerID >= 0"}
    results = collection.search(**search_param)
    searchResults = []
    for i, result in enumerate(results):
        temp = []
        #print("\nSearch result for {}th vector: ".format(i))
        for j, res in enumerate(result):
            if res.id == id:
                pass
                #continue
            #print("Top {}: {}".format(j, res))
            temp.append((res.id, res.distance))
            #print(i, res.id)
        searchResults.append(temp)
    return searchResults#print(data[i][0], data[res.id][0])

def ModifiedSearch(collection, vector_field, id_field, search_vectors):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
        "limit": _TOPK,
        "expr": "id_field >= 0"}
    results = collection.search(**search_param)
    for i, result in enumerate(results):
        print("\nSearch result for {}th vector: ".format(i))
        for j, res in enumerate(result):
            print("Top {}: {}".format(j, res))
            #print(i, res.id)
            #print(data[i][0], data[res.id][0])


def set_properties(collection):
    collection.set_properties(properties={"collection.ttl.seconds": 1800})


def main():
    # create a connection
    create_connection()

    # drop collection if the collection exists
    if has_collection(_COLLECTION_NAME_QUESTION):
        drop_collection(_COLLECTION_NAME_QUESTION)
    if has_collection(_COLLECTION_NAME_ANSWER):
        drop_collection(_COLLECTION_NAME_ANSWER)

    # create collection
    collectionQuestion = create_collection(_COLLECTION_NAME_QUESTION, _ID_FIELD_NAME_QUESTION, _VECTOR_FIELD_NAME_QUESTION)
    collectionAnswer = create_collection(_COLLECTION_NAME_ANSWER, _ID_FIELD_NAME_ANSWER, _VECTOR_FIELD_NAME_ANSWER)


    # alter ttl properties of collection level
    set_properties(collectionQuestion)
    set_properties(collectionAnswer)

    # show collections
    list_collections()

    # insert 10000 vectors with 128 dimension
    vectorsQuestion = insert(collectionQuestion, 10000, _DIM)
    vectorsAnswer = insert(collectionAnswer, 10000, _DIM)

    collectionAnswer.flush()
    collectionQuestion.flush()


    # get the number of entities
    get_entity_num(collectionQuestion)
    get_entity_num(collectionAnswer)

    # create index
    create_index(collectionQuestion, _VECTOR_FIELD_NAME_QUESTION)
    create_index(collectionAnswer, _VECTOR_FIELD_NAME_ANSWER)

    # load data to memory
    load_collection(collectionQuestion)
    load_collection(collectionAnswer)


    # search
    id = 120
    print("Query: ", data[id][0])
    query = vectorsQuestion[id]
    print("------Real Answer------")
    for x in finalEmbeddings_[id]:
        print(x[0], end="; ")
    print("\n")
    for x in finalEmbeddings_[id]:
        print(flatData[x[0]])
    print("\n")

    qs = search(collectionQuestion, _VECTOR_FIELD_NAME_QUESTION, _ID_FIELD_NAME_QUESTION, query, id)
    qsss = search(collectionAnswer, _VECTOR_FIELD_NAME_ANSWER, _ID_FIELD_NAME_ANSWER, query, id)
    print("WAY 1: ", qsss)
    for item in qsss[0]:
        print(flatData[item[0]])
    newSearchQuery = []
    for i in qs:
        for j in i:
            newSearchQuery.append(vectorsQuestion[j[0]])
    qss = search(collectionAnswer, _VECTOR_FIELD_NAME_ANSWER, _ID_FIELD_NAME_ANSWER, newSearchQuery, id)
    #for x in qss:
        #print(x)
    combinedResult = []
    for i in range(len(qs)):
        for j in range(len(qs[i])):
            for k in range(len(qss[j])):
                #print(qss[j][k], qs[i][j])
                temp = list(qss[j][k])
                # Perform the assignment
                temp[1] = temp[1] + qs[i][j][1]
                # Convert the list back to a tuple
                qss[j][k] = tuple(temp)
                combinedResult.append(qss[j][k])
    combinedResult = sorted(combinedResult, key=lambda x: x[1])
    print("\n\n\nWAY 2: ", combinedResult)
    for item in combinedResult:
        print(flatData[item[0]])
    potentialAnswers = []
    for i in qs:
        for j in i:
            potentialAnswers.append((j[0], finalEmbeddings_[j[0]]))
    results = []
    for item in potentialAnswers:
        temp = []
        for j in range(len(item[1])):
            temp.append((item[1][j][0], np.linalg.norm(np.array(query) - np.array(item[1][j][1]))))
        results.append(temp)

    similarAnswers = []
    for item in results:
        for element in item:
            similarAnswers.append(element)
    #print(similarAnswers)
    similarAnswers = sorted(similarAnswers, key=lambda x: x[1])
    print("\n\n\nWAY 3: ", similarAnswers)
    for item in similarAnswers:
        print(flatData[item[0]])

    # release memory
    release_collection(collectionQuestion)
    release_collection(collectionAnswer)

    # drop collection index
    drop_index(collectionQuestion)
    drop_index(collectionAnswer)

    # drop collection
    drop_collection(_COLLECTION_NAME_QUESTION)
    drop_collection(_COLLECTION_NAME_ANSWER)


if __name__ == '__main__':
    main()