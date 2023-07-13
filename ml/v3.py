import os
import sys
from multiprocessing import Pool, cpu_count
from typing import List, Optional
import readline
import atexit
from langchain import LLMChain
from langchain.chains import AnalyzeDocumentChain, ReduceDocumentsChain, StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from loguru import logger
from langchain.memory import ConversationBufferMemory


class FileProcessor:
    def __init__(self, directory_path):
        self.path = directory_path
        self.processed_files = None
        self.memory = ConversationBufferMemory()

    @staticmethod
    def read_file(file_path: str) -> Optional[str]:
        try:
            with open(file_path, "r") as file:
                return file.read()
        except IOError as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
        return None

    def process_file(self, file_path: str) -> List[Document]:
        if content := self.read_file(file_path):
            text_splitter = CharacterTextSplitter()
            content = text_splitter.split_text(content)
            return [Document(page_content=t) for t in content[:3]]
        else:
            return []

    def process_directory(self):
        if self.processed_files is None:
            files = [
                os.path.join(root, file)
                for root, _, files in os.walk(self.path)
                for file in files
                if file.split(".")[-1] in [
                    "py", "java", "js", "ts", 'tsx', "c", "cpp", "h", "cs", "go", "rs", "php",
                    "html", 'jsx,', 'json', 'css', 'scss', 'sass', 'less', 'vue', 'swift',
                    'kt', 'ktm', 'kts', 'ktm', 'ktx', 'kts', 'ktm', 'ktx', 'kts', 'ktm',
                ]
            ]

            if not files:
                logger.info("No Python files found in the directory.")
                return []

            num_workers = min(cpu_count(), len(files)) if files else 1
            with Pool(num_workers) as p:
                results = p.map(self.process_file, files)

            self.processed_files = [doc for sublist in results for doc in sublist]

        logger.info(f"Processed {len(self.processed_files)} files.")
        return self.processed_files

    # Function to generate refactor plan and run analysis
    def generate_refactor_plan(self, processed_files: list) -> None:
        models = ["gpt-4-0613", "gpt-3.5-turbo-16k"]
        llm = ChatOpenAI(temperature=0.7, model_name=models[1])  # # LLM setup
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
        )

        # Create a documents' chain with a prompt to summarize code structure
        stack_chain: StuffDocumentsChain = self.generate_plan(
            "given the context {context} - summarize all functions, classes, methods. give an ideal tree view of the best structure for this application and be as detailed as possible about the code use and flow - each response should be at least a paragraph if possible."
            ,
            llm,
            document_prompt,
            'context'
        )
        chain = ReduceDocumentsChain(
            combine_documents_chain=stack_chain,
        )
        try:
            # Run analysis on the created chain and processed files
            self.run_analysis(chain, processed_files)
        except Exception as e:
            logger.error(f"An error occurred while generating the refactor plan for a document: {str(e)}")

    # Function to create a plan for documents' chain
    def generate_plan(
        self, template: str, llm: ChatOpenAI,
        document_prompt: PromptTemplate, document_key: Optional[str] = None
    ) -> StuffDocumentsChain:
        prompt_combine = PromptTemplate(
            input_variables=document_key and [document_key] or [],
            template=template
        )
        llm_chain_combine = LLMChain(llm=llm, prompt=prompt_combine)
        return StuffDocumentsChain(
            llm_chain=llm_chain_combine,
            document_prompt=document_prompt,
            document_variable_name=document_key or "page_content",

        )

    # Function to run analysis on documents' chain & interact with user
    def run_analysis(self, chain: ReduceDocumentsChain, processed_files: list) -> None:
        analysis_agent = AnalyzeDocumentChain(combine_docs_chain=chain, input_key="page_content")
        analysis = analysis_agent.run(chain.combine_docs(processed_files, token_max=16000)[0])  # Run the agent & get
        # analysis result

        logger.info(analysis)

        # Call function to interact with user using analyzed information
        self.interact(analysis)

    # Function to interact with user using AI chat model
    def interact(self, analysis: str) -> None:
        llm: 'ChatOpenAI' = ChatOpenAI(temperature=0.7, model_name="gpt-4-0613")  # Chat model setup
        while True:
            custom_prompt: str = input("Enter your question or 'exit' to quit: ")
            if custom_prompt.lower() == 'exit':
                break

            # Assuming that the 'analysis' contains the context for llm to respond to
            prompt = PromptTemplate(
                input_variables=["context"],
                template="{context}, Given the context " + analysis
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            response: str = llm_chain.run(custom_prompt)
            logger.info(f"Response: {response}")

def main(directory_path):
    dp = FileProcessor(directory_path)
    while True:
        print("Menu:")
        print("1. Scan directory")
        print("2. Generate refactor plan")
        print("3. Exit")
        choice = input("Enter your choice: ")
        if choice == '1':
            dp.process_directory()
        elif choice == '2':
            if processed_files := dp.process_directory():
                # stom_prompt = input("Enter a custom prompt: ")
                dp.generate_refactor_plan(processed_files)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please choose again.")

import contextlib


if __name__ == "__main__":
    logger.configure(
        handlers=[
            dict(
                sink='app.log', level="DEBUG", format="{time} {level} {message}",
                colorize=True, backtrace=True, diagnose=True
            ),
            dict(
                sink=sys.stdout, level="INFO", format="{time} {level} {message}",
                colorize=True, backtrace=True, diagnose=True
            ),
        ],
    )

    histfile = os.path.join(os.path.expanduser("~"), ".python_history")
    with contextlib.suppress(FileNotFoundError):
        readline.read_history_file(histfile)
    atexit.register(readline.write_history_file, histfile)

    directory_path = input("Enter directory path: ")
    main(directory_path)
