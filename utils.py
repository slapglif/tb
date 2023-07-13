import os
import json
import openai
from transformers import BertTokenizer

# Define the directory
directory = r"C:\Users\freeb\work\tradebot"

# Define the output JSON file
output_json = "output.json"

# Initialize BertTokenizer for subword tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define the chunk size
chunk_size = 512


def chunk_file(file_content):
    tokens = tokenizer.tokenize(file_content)
    for i in range(0, len(tokens), chunk_size):
        yield " ".join(tokens[i : i + chunk_size])


def process_file(file_path):
    with open(file_path, "r", errors="replace") as file:
        content = file.read()
    for chunk in chunk_file(content):
        yield {"content": chunk}


def summarize_files_in_directory():
    output = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                try:
                    for json_content in process_file(file_path):
                        output.append(json_content)
                except Exception as e:
                    print(f"Failed to process file: {file_path}. Error: {str(e)}")

    with open(output_json, "w") as json_file:
        json.dump(output, json_file)


def get_refactor_plan():
    with open(output_json, "r") as read_file:
        data = json.load(read_file)

    # Prepare the context for GPT-4
    context = "Here are summaries of the contents of multiple Python files:\n\n"
    for item in data:
        context += "- File Content: " + item["content"] + "\n\n"

    context += "Given these file contents, please provide a refactor and restructure plan based on the available functions, classes methods, use case, flow, and intended new file structure."

    # Make request to OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-002", prompt=context, max_tokens=256
    )

    # Extract the refactor plan
    refactor_plan = response.choices[0].text.strip()

    return refactor_plan


def main():
    summarize_files_in_directory()
    refactor_plan = get_refactor_plan()
    print(refactor_plan)


if __name__ == "__main__":
    main()
