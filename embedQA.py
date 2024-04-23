from nltk.tokenize import sent_tokenize
import tiktoken
import os
import boto3
import json
import pickle

import base64
from botocore.exceptions import ClientError

bedrock = boto3.client(service_name="bedrock-runtime",region_name='us-east-1')

modelId = "cohere.embed-english-v3"

accept = "application/json"
contentType = "application/json"


GPT_DEPLOYMENT_NAME="gpt-4-turbo"
encoding = tiktoken.encoding_for_model(GPT_DEPLOYMENT_NAME)

with (open("QAGPT4.pickle", "rb") as f):
    responses, totalToken, elapsed_time = pickle.load(f)

results = []
totalOutToken = 0
for response in responses:
    # Parse the JSON content
    response = response.content.replace('```json\n', '').replace('\n```', '')
    content_json = json.loads(response)
    # Extract the "Analysis" part
    analysis = content_json["Analysis"]
    if isinstance(analysis, list):
        # Check if all elements in the list are strings
        if all(isinstance(elem, str) for elem in analysis):
            # Concatenate all strings
            analysis = ''.join(analysis)
    totalOutToken += len(encoding.encode(analysis))
    analysis = sent_tokenize(analysis)
    results.append(analysis)


with open("whquestions.pickle", "rb") as f:
    _, wh_questions, _ = pickle.load(f)

data = []
for i in range(len(wh_questions)):
    print("Question: ", wh_questions[i])
    print("Answer: ", results[i])
    temp = []
    temp.append(wh_questions[i])
    temp += results[i]
    data.append(temp)
    print("\n")


print(f"Elapsed time: {elapsed_time} seconds")
print("Total Input Tokens: ", totalToken)
print("Total Output Tokens: ", totalOutToken)


# Create the AWS client for the Bedrock runtime with boto3
aws_client = boto3.client(service_name="bedrock-runtime",region_name='us-east-1')

# Input parameters for embed. In this example we are embedding hacker news post titles.

if os.path.exists("QAEmbeddings.pickle"):
    with open("QAEmbeddings.pickle", "rb") as f:
        finalEmbeddings, data = pickle.load(f)
else:
    finalEmbeddings = []
    for i in range(len(data)):
        print(i)
        input_type = "clustering"
        truncate = "NONE" # optional
        model_id = "cohere.embed-english-v3" # or "cohere.embed-multilingual-v3"

        # Create the JSON payload for the request
        json_params = {
                'texts': data[i],
                'truncate': truncate,
                "input_type": input_type
            }
        json_body = json.dumps(json_params)
        params = {'body': json_body, 'modelId': model_id,}

        # Invoke the model and print the response
        result = aws_client.invoke_model(**params)
        response = json.loads(result['body'].read().decode())
        finalEmbeddings.append(response['embeddings'])

    with open("QAEmbeddings.pickle", "wb") as f:
        pickle.dump([finalEmbeddings, data], f)