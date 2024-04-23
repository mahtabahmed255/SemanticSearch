import sys

# Backup the original stdout
original_stdout = sys.stdout

# Open your file in write mode
with open('/Users/mahtabahmed/output_handpicked.txt', 'w') as file:
    sys.stdout = file  # Redirect stdout to your file
    tempList = [99996, 99475, 100001, 100022, 100045, 100216, 100234, 100238, 100248, 102591, 104067, 90264, 90281,
                92697, 96037, 84613, 84990]
    # Open the file in read mode
    with open('/Users/mahtabahmed/rt_output_1.txt', 'r') as read_file:
        # Skip the first line
        next(read_file)
        # Read each line in the file
        for line in read_file:
            # Split the line by TAB
            elements = line.split('\t')
            # Assign the elements to variables
            ID, title, conversation = elements[0], elements[1], elements[2]
            if int(ID) in tempList:
            # Print each sample with newline characters
                print(f"ID: {ID}\nTitle: {title}\nConversation: {conversation.replace('\\n', '\n')}\n")
                print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")

    sys.stdout = original_stdout  # Reset stdout back to its original value