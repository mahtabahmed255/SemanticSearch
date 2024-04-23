from collections import defaultdict

with open("/Users/mahtabahmed/unique_titles.txt", 'r') as file:
    lines = file.readlines()
# Input lines from grep command

# Function to check if two lines have only one word difference
def one_word_difference(line1, line2):
    words1 = line1.split()
    words2 = line2.split()
    if len(words1) != len(words2):
        return False
    diff_count = sum(1 for w1, w2 in zip(words1, words2) if w1 != w2)
    return diff_count == 1

# Group lines based on similarity
groups = defaultdict(list)
for line in lines:
    found_group = False
    for group_lines in groups.values():
        for group_line in group_lines:
            if one_word_difference(line, group_line):
                group_lines.append(line)
                found_group = True
                break
        if found_group:
            break
    if not found_group:
        groups[line] = [line]

sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

# Print the grouped lines
for i, (group, lines) in enumerate(sorted_groups, 1):
    print(f"Group {i}:")
    for line in lines:
        print(line)
