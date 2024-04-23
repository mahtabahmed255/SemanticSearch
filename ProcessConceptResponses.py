import sys

import pickle

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

with open("gpt4Responses/nonQAGPT4.pickle", "rb") as f:
    responses, totalInToken, totalOutToken, elapsed_time = pickle.load(f)
print(len(responses), totalInToken, totalOutToken)
keepIdList = [item for item in idss if item not in filteredIDs]

r1 = responses[0:3251]
r2 = responses[3251:]

r1_ = []
for i in range(0, len(r1)):
    if idss[i] not in filteredIDs:
        r1_.append(r1[i])

print(len(r1_) + len(r2), len(idss) - len(filteredIDs), len(keepIdList))

finalResponses = r1_ + r2
with open("gpt4Responses/nonQAGPT4_final.pickle", "wb") as f:
    pickle.dump([finalResponses, keepIdList], f)

