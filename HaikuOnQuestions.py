import boto3
import json
import base64
import pickle
from botocore.exceptions import ClientError

bedrock = boto3.client(service_name="bedrock-runtime",region_name='us-east-1')

modelId = "anthropic.claude-3-haiku-20240307-v1:0"

accept = "application/json"
contentType = "application/json"

file_path = "/Users/mahtabahmed/rt_output.txt"  # Replace "your_file.txt" with the actual file path
with open(file_path, 'r') as file:
    lines = file.readlines()

questions = []
responses = []
for line in lines:
    line = line.split("\t")
    if (len(line) == 3):
        questions.append(line[1])
        responses.append(line[2])
print(len(questions))

with open("whquestions.pickle", "rb") as f:
    wh_question_embeddings, wh_questions, ids = pickle.load(f)

j = 0
results = []
for i in ids:
    question = questions[i]
    conversation = responses[i]
    prompt = """I am building a question-answer dataset. For each ticket, I have the title of that ticket and the conversation history between support team and the customer who raises the ticket in Jira.
    I want you to investigate the conversation history of each ticket and provide me the list of steps suggested by the developer to solve the problem mentioned in the title. Make each step coherent and grammatically correct so that I can run semantic search with them.
    Do not use list of numbers.
    Apart from the instruction above do not include any additional text. Just return the following structure:
    Analysis - {}
    I really need this to work well so that I can keep my job! I repeat again, do not return anything except JSON. here is the ticket data:
    """
    data = "Issue title: {}. conversation history: {}."
    data = data.format(question, conversation)
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt + data
                    },
                ],
            }
        ],
    }

    try:
        response = bedrock.invoke_model(
            modelId=modelId,
            body=json.dumps(request_body),
        )

        # Process and print the response
        result = json.loads(response.get("body").read())
        input_tokens = result["usage"]["input_tokens"]
        output_tokens = result["usage"]["output_tokens"]
        output_list = result.get("content", [])

        # print("Invocation details:")
        # print(f"- The input length is {input_tokens} tokens.")
        # print(f"- The output length is {output_tokens} tokens.")

        # print(f"- The model returned {len(output_list)} response(s):")
        print("Question: ", question)
        print("Response: ", conversation)
        print("Response Generated: ", end=" ")
        for output in output_list:
            print(output["text"])

    except ClientError as err:
        print(
            "Couldn't invoke Claude 3 Haiku Vision. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise
    except Exception as err:
        print("Couldn't invoke Claude 3 Haiku Vision. Here's why: %s", err)
        raise
    j += 1
    if j == 10:
        break
