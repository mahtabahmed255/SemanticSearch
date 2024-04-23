from openai import OpenAI
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle


# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Read the file
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

results = []
for i in range(len(questions)):
    question = questions[i]
    conversation = responses[i]
    #prompt = "I am building a question-answer dataset. For each ticket, I have the title of that ticket and the conversation history between support team and the customer who raises the ticket in Jira. I want you to analyze each ticket and identify, if the ticket title conveys as a fact based question. Do not return any additional text. Just return the following structure: flag : {Does the title represents a fact based question (TRUE/FALSE)}. title : {IF falg is TRUE then the rewritten title ELSE the original title}. I really need this to work well so that I can keep my job! I repeat again, do not return anything except JSON. here is a sample data:"
    prompt = text = """I am building a question-answer dataset. For each ticket, I have the title of that ticket and the conversation history between support team and the customer who raises the ticket in Jira.
I want you to analyze each ticket and identify, if the conversation history leads to some kind of solution or not. Basically you need to identify whether the conversation history talks about a solution to a problem in the title or not. You also need to analyze the conversation history to highlight the final solution. The analysis structure should be DO THIS or DO THAT and should not include any named entities. Do not return any additional text. Just return the following structure:
flag : {if the conversation history talks about a solution to a problem in the title or not (TRUE/FALSE)}
Analysis - {ANALYSIS OF conversation history GOES HERE}  I really need this to work well so that I can keep my job! I repeat again, do not return anything except JSON. here is a sample data:
"""


    data = "Issue title: {}. conversation history: {}."
    data = data.format(question, conversation)

    history = [
        {"role": "system",
         "content": "Welcome to Jira Issue analyzer! Please describe the issue you want me to analyze.",},
        {"role": "user",
         "content": prompt + data},
    ]
    print(prompt + data)

    completion = client.chat.completions.create(
        model="local-model",  # this field is currently unused
        messages=history,
        temperature=0.7,
        stream=False,
    )
    results.append (completion.choices[0].message.content)
    print(i, results[-1])