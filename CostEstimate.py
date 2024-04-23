from transformers import GPT2TokenizerFast
import tiktoken
import json
import sys
import re
import pickle

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
pattern1, pattern2, pattern3, pattern4 = [], [], [], []

# Backup the original stdout
original_stdout = sys.stdout

GPT_DEPLOYMENT_NAME="gpt-4-turbo"
encoding = tiktoken.encoding_for_model(GPT_DEPLOYMENT_NAME)
#totalInToken += len(encoding.encode(nonQAPrompt + data))

# CLAUDE
tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/claude-tokenizer')
assert tokenizer.encode('hello world') == [9381, 2253]

pattern = r"New\s+SAP\s+organization\s+\S+\s+created"
temp = []
with open('/Users/mahtabahmed/rt_output_1.txt', 'r') as read_file:
    # Skip the first line
    next(read_file)
    # Read each line in the file
    for line in read_file:
        # Split the line by TAB
        elements = line.split('\t')
        # Assign the elements to variables
        ID, title, conversation = elements[0], elements[1], elements[2]
        if "NULL" in conversation:
            pattern1.append(ID)
            continue
        if re.match(pattern, title):
            pattern2.append(ID)
            continue
        if "No real alert" in conversation:
            pattern3.append(ID)
            temp.append(conversation)
            continue
        if "opsgenie@opsgenie.net wrote:" in conversation:
            pattern4.append(ID)
            continue

print("pattern1: ", len(pattern1))
print("pattern2: ", len(pattern2))
print("pattern3: ", len(pattern3))
print("pattern4: ", len(pattern4))

total = set(pattern1 + pattern2 + pattern3 + pattern4)
total = [int(item) for item in total]

with open("/Users/mahtabahmed/filteredIDs.pickle", "wb") as file:
    pickle.dump(total, file)

print("Total: ", len(pattern1) + len(pattern2) + len(pattern3) + len(pattern4), "-->", len(total))
claudeInToken = 0
gptInToken = 0
with open('/Users/mahtabahmed/rt_output_1.txt', 'r') as read_file:
    # Skip the first line
    next(read_file)
    # Read each line in the file
    for line in read_file:
        # Split the line by TAB
        elements = line.split('\t')
        # Assign the elements to variables
        ID, title, conversation = elements[0], elements[1], elements[2]
        if ID not in total:
            #CLaude estimate
            data = conceptPrompt + title + conversation
            claudeInToken += len(tokenizer.encode(data))
            gptInToken += len(encoding.encode(data))
            temp.append(ID)


# Open the file in read mode
with open('/Users/mahtabahmed/gptout.txt', 'r') as file:
    # Read the entire file
    gptOut = file.read()
with open('/Users/mahtabahmed/sonnetout.txt', 'r') as file:
    # Read the entire file
    sonnetOut = file.read()

claudeOutToken = (len(tokenizer.encode(sonnetOut))/ 17) * len(temp)
gptOutToken = (len(encoding.encode(gptOut))/ 17) * len(temp)

# Now, 'content' contains the entire file as a string

print("Total Issues: ", len(temp))
gptInPrice = 10 # 10 USD for 1 million
gptOutPrice = 30 # 30 USD for 1 million
claudeInPrice = .003 # .003 USD for 1000
claudeOutPrice = .015 # .015 USD for 1000
print("Claude In Token: ", claudeInToken, " Cost: ", claudeInToken/1000 * 0.003, " USD")
print("gpt In Token: ", gptInToken, " Cost: ", gptInToken/1000000 * 10, " USD")
print("Claude Out Token: ", claudeOutToken, " Cost: ", claudeOutToken/1000 * 0.015, " USD")
print("gpt Out Token: ", gptOutToken, " Cost: ", gptOutToken/1000000 * 30, " USD")