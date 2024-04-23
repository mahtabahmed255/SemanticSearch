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

import re


def identify_wh_questions(sentences):
    wh_questions = []
    # Regular expression to match WH questions
    wh_question_pattern = r'\b(who|what|when|where|why|which|how|whose|whom|whichever|whatever|whenever|wherever|whyever|howsoever|how|what)\b.*\?'
    i = 0
    ids = []
    for sentence in sentences:
        # Check if the sentence matches the WH question pattern
        if re.match(wh_question_pattern, sentence, re.IGNORECASE):
            wh_questions.append(sentence)
            ids.append(i)
        i += 1
    return wh_questions, ids



# Identify WH questions
wh_questions, ids = identify_wh_questions(questions)
wh_responses = [responses[i] for i in ids]

# Print identified WH questions
print("Identified WH questions:")
for question in wh_questions:
    print(question)

print(len(wh_questions))
for res in wh_responses:
    print(res)
    print("\n")

from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
embeddings = HuggingFaceEmbeddings()
wh_question_embeddings = []
for question in wh_questions:
    query_result = embeddings.embed_query(question)
    print(query_result)
    wh_question_embeddings.append(query_result)
with open("whquestions.pickle", "wb") as f:
    pickle.dump([wh_question_embeddings, wh_questions, ids], f)
