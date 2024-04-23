import boto3
import json
import base64
import pickle
from botocore.exceptions import ClientError
import time

bedrock = boto3.client(service_name="bedrock-runtime",region_name='us-east-1')

modelId = "anthropic.claude-3-sonnet-20240229-v1:0"

accept = "application/json"
contentType = "application/json"


file_path = "/Users/mahtabahmed/rt_output_1.txt"  # Replace "your_file.txt" with the actual file path
with open(file_path, 'r') as file:
    lines = file.readlines()

questions = []
responses = []
idss = []
for line in lines[1:]:
    line = line.split("\t")
    if (len(line) == 3):
        idss.append(int(line[0]))
        questions.append(line[1])
        responses.append(line[2])
print(len(questions))

with open("whquestions.pickle", "rb") as f:
    wh_question_embeddings, wh_questions, ids = pickle.load(f)

conceptPrompt = """I am building a semantically searchable database for our bug tracking system so that we can get high quality 
search results and similarity matching. To do this we are creating embeddings for the information. 
However, the tickets contain a lot of unnecessary information and noise that makes it hard to get good 
semantic search results. 
I want you to read the issue report copied below and to generate a list of information that would 
capture the essence of the issue. Each line that you provide will be turned into an embedding 
and placed in a vector database.
Information about the logistics of the issue are not important. For example, do not include any statements 
on which sprint the issue is in or who did what. Do not include information about the effort to fix. 
Only mention references to other issues if there is sufficient context in the statement to make a semantic 
search possible.
Remember that each line you produce will be embedded independently and must have enough semantic qualities to provide meaningful matches. I just want to capture unique semantic qualities of the issue. 
For concepts you choose, be verbose to fully capture the concept. Focus on the technical pieces of the issue. Generic comments that could apply to any bug (such as: "fix has been implemented in 
branch X") are not useful for semantic searches and should be skipped
If the issue is very trivial or has no technical content, then you can simply return an empty list for the concepts. If the issue is very long, then you can return a list of concepts that are not exhaustive. 
The only output that you should return is the list as a YAML array of single strings and a succinct summary field as shown below: 
<yaml-response-schema>
summary: "<up to 6 word summary-text>"
concepts:
- concept-1
- concept-2
...
</yaml-response-schema>

I really need this to work well so that I can keep my job!
Here is the issue report:
"""

QAPrompt = """I am building a question-answer dataset. For each ticket, I have the title of that ticket and the conversation history between the developer in the support team and the customer who raises the ticket in Jira.
I want you to analyze the conversation history and provide me the list of steps suggested by the developer to solve the problem mentioned in the title. Make each step coherent and grammatically correct so that I can run semantic search with them.
Do not use list of numbers.
Apart from the instruction above, do not return any additional text. Just return the following structure:
Analysis - {STEPS SUGGESTED BY THE DEVELOPER}
I really need this to work well so that I can keep my job! I repeat again, do not return anything except JSON. Here is the ticket data:
"""

nonQAPrompt = """I am building a question-answer dataset. For each ticket, I have the title of that ticket and the conversation history between the developer in the support team and the customer who raises the ticket in Jira.
I want you to analyze the title and conversation history and rewrite the title with proper natural language structure in a way that it has the problem statement discussed in the conversation history. If the rewriting is not possible return the original title.
I want you to analyze the conversation history and provide me the list of steps suggested by the developer to solve the problem mentioned in the title. Make each step coherent and grammatically correct so that I can run semantic search with them. If the conversation history does not have enough content, return an empty list.
Do not use list of numbers.
Apart from the instruction above, do not return any additional text. Just return the following structure:
Title - {REWRITTEN TITLE}
Analysis - {STEPS SUGGESTED BY THE DEVELOPER}
I really need this to work well so that I can keep my job! I repeat again, do not return anything except JSON. Here is the ticket data:
"""

results = []
tempList = [99996, 99475, 100001, 100022, 100045, 100216, 100234, 100238, 100248, 102591, 104067, 90264, 90281, 92697, 96037, 84613, 84990]
# Start timing
start_time = time.time()
totalInToken = 0
totalOutToken = 0
for i in range(len(questions)):
    if i in ids:
        continue
    if idss[i] not in tempList:
        continue
    #print(responses[i])
    question = questions[i]
    conversation = responses[i]
    data = "Issue title: {}. conversation history: {}."
    data = data.format(question, conversation)
    #totalInToken += len(encoding.encode(nonQAPrompt + data))
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": nonQAPrompt + data
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
        for output in output_list:
            print(output["text"])
        #print("\n", data)


        print("ID: ", idss[i])


    except ClientError as err:
        print(
            "Couldn't invoke Claude 3 Sonnet Vision. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise
    except Exception as err:
        print("Couldn't invoke Claude 3 Haiku Vision. Here's why: %s", err)
        raise

# End timing
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

with open("nonQASonnet.pickle", "wb") as f:
    pickle.dump([results, totalInToken, totalOutToken, elapsed_time], f)
