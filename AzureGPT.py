'''
1. Reads rt_output_N.txt file where N represents last N years of data
2. Read filteredIDs.pickle where filtered Ids have the ids that are curated base on four different regular expression pattern in the data
3. Run GPT4 model with conceptPrompt on the filtered Data
4. Store the result
'''

from langchain_community.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
import os
import pickle
import time
import tiktoken
import json

GPT_DEPLOYMENT_NAME="gpt-4-turbo"
encoding = tiktoken.encoding_for_model(GPT_DEPLOYMENT_NAME)

os.environ["AZURE_OPENAI_API_KEY"] = "9605d652498e452cb620c7b3a271c4c6"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ed-openai-test-canada-east.openai.azure.com"

model = AzureChatOpenAI (
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version="2023-03-15-preview",
    azure_deployment=GPT_DEPLOYMENT_NAME,
)

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
print(len(questions), len(questions), len(idss))

with open("/Users/mahtabahmed/filteredIDs.pickle", "rb") as f:
    filteredIDs = pickle.load(f)

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

# Start timing
start_time = time.time()
totalInToken = 0
totalOutToken = 0
import pickle
with open("gpt4Responses/nonQAGPT4_12750.pickle", "rb") as f:
    results, totalInToken, totalOutToken= pickle.load(f)

for i in range(12751, len(questions)): #3251 12750
    if idss[i] in filteredIDs:
        continue
    #if i in ids:
        #continue
    #if idss[i] != 85990:
        #continue
    #print(responses[i])
    question = questions[i]
    conversation = responses[i]
    data = "\nIssue title: {}. \nConversation History: {}"

    totalInToken += len(encoding.encode(nonQAPrompt + data))
    message = HumanMessage(
        content = conceptPrompt + data
    )
    content = model([message]).content.replace('```json\n', '').replace('\n```', '')
    #json_object = json.loads(content)
    #print(data)
    #print("Generated Title: ", json_object["Title"], "\nGenerated Analysis: ", json_object["Analysis"])
    totalOutToken += len(encoding.encode(model([message]).content))
    results.append(content)
    #print(str(len(results)) + "|" + str(len(ids)) + " " + str(model([message])))
    #print("\n")
    print(len(results))
    if i % 50 == 0:
        with open("gpt4Responses/nonQAGPT4_" + str(i) + ".pickle", "wb") as f:
            pickle.dump([results, totalInToken, totalOutToken], f)

# End timing
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
print("Total Input Tokens: ", totalInToken)
print("Total Output Tokens: ", totalOutToken)

with open("gpt4Responses/nonQAGPT4.pickle", "wb") as f:
    pickle.dump([results, totalInToken, totalOutToken, elapsed_time], f)