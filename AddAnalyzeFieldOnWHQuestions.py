# Chat with an intelligent assistant in your terminal
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

results = []
for i in range(len(wh_questions)):
    question = wh_questions[i]
    conversation = wh_responses[i]
    prompt = "I am building a semantically searchable database for our bug tracking system so that we can get high quality search results and similarity matching. To do this we are creating embeddings for the information. However, the tickets contain a lot of unnecessary information and noise that makes it hard to get good semantic search results. I want you to read the issue report copied below and to generate a list of information that would capture the essence of the issue. Each line that you provide will be turned into an embedding and placed in a vector database. Information about the logistics of the issue are not important. For example, do not include any statements on which sprint the issue is in or who did what. Do not include information about the effort to fix. Only mention references to other issues if there is sufficient context in the statement to make a semantic search possible. Remember that each line you produce will be embedded independently and must have enough semantic qualities to provide meaningful matches. I just want to capture unique semantic qualities of the issue. For concepts you choose, be verbose to fully capture the concept. Focus on the technical pieces of the issue. Generic comments that could apply to any bug (such as: 'fix has been implemented in branch X') are not useful for semantic searches and should be skipped. If the issue is very trivial or has no technical content, then you can simply return an empty list for the concepts. If the issue is very long, then you can return a list of concepts that are not exhaustive. The only output that you should return the list as a YAML array of single strings, followed by an analysis field which talks about how the issue is resolved and a succinct summary field as shown below: <yaml-response-schema> summary: '<up to 6 word summary-text>' concepts: - concept-1 - concept-2 ... Analysis: '<be coherent and try to capture the take a way of the conversation>' </yaml-response-schema> In the analysis field, do not talk about what the issue is, just highlight the final solution. The analysis structure should be DO THIS or DO THAT and should not include any named entities. I really need this to work well so that I can keep my job! Here is the issue report:"
    data = "Issue title: {}. Conversation between support team and Developer: {}. Remember, provide me just summary and concepts. No additional text please."
    data = data.format(question, conversation)

    history = [
        {"role": "system",
         "content": "Welcome to BugAnalyzer! Please describe the issue you want me to analyze.",},
        {"role": "user",
         "content": prompt + data},
    ]

    completion = client.chat.completions.create(
        model="local-model",  # this field is currently unused
        messages=history,
        temperature=0.7,
        stream=False,
    )
    results.append (completion.choices[0].message.content)
    print(i, results[-1])

with open("results.pickle", "wb") as f:
    pickle.dump(results, f)


embeddings = HuggingFaceEmbeddings()
wh_question_embeddings = []
for question in wh_questions:
    query_result = embeddings.embed_query(question)
    print(query_result)
    wh_question_embeddings.append(query_result)
with open("whquestions.pickle", "wb") as f:
    pickle.dump(wh_question_embeddings, f)





