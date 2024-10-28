#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd Desktop


# In[2]:


import os
import pandas as pd
from Bio import SeqIO
import chardet
import json
import PyPDF2
import speech_recognition as sr
import pyttsx3
import threading


# In[3]:


import os
import openai

# Set your OpenAI API key here
openai.api_key = 'Your API KEY HERE'


# In[4]:


def read_bioinformatics_file(file_path):
    """
    Reads a bioinformatics file and returns the data in a suitable format.
    
    Parameters:
        file_path (str): Path to the bioinformatics file.
    
    Returns:
        data: Parsed data from the file.
    """
    file_extension = os.path.splitext(file_path)[-1].lower()
    
    try:
        if file_extension in ['.pdf']:
            return read_pdf_file(file_path)
        # Handling Genomics/Metagenomics files
        if file_extension in ['.fasta', '.fa']:
            data = list(SeqIO.parse(file_path, 'fasta'))
        elif file_extension in ['.fastq']:
            data = list(SeqIO.parse(file_path, 'fastq'))
        elif file_extension in ['.vcf']:
            # Reading VCF file while ignoring metadata (lines starting with #)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                data_lines = [line.strip() for line in lines if not line.startswith('#')]
            
            # Converting data lines into a DataFrame for structured analysis
            data = pd.DataFrame([line.split('\t') for line in data_lines],
                                columns=['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'])
        
        elif file_extension in ['.gff', '.gtf']:
            data = pd.read_csv(file_path, comment='#', delimiter='\t', header=None)
            data.columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
        elif file_extension in ['.biom']:
            with open(file_path, 'r') as f:
                data = json.load(f)  # BIOM format is often in JSON
        elif file_extension in ['.kraken']:
            data = pd.read_csv(file_path, delimiter='\t', header=None, names=['classification', 'read_id', 'taxonomy', 'length', 'kmer_score'])
        
        # Handling Proteomics and other files
        elif file_extension in ['.mgf']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = f.readlines()  # MGF is a text-based format
        elif file_extension in ['.mzml']:
            data = pd.read_xml(file_path)  # mzML is often in XML format, so we parse it as such
        elif file_extension in ['.csv']:
            data = pd.read_csv(file_path)  # General use for abundance data and other tabular formats
        elif file_extension in ['.tsv']:
            data = pd.read_csv(file_path, delimiter='\t')  # TSV format for tab-separated values
        elif file_extension in ['.xlsx']:
            data = pd.read_excel(file_path)  # Proper handling of Excel file format
        
        # Handling other text-based file formats with encoding detection
        else:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                detected_encoding = result['encoding']

            with open(file_path, 'r', encoding=detected_encoding, errors='ignore') as file:
                data = file.read()
    
    except UnicodeDecodeError as e:
        print(f"Error reading file {file_path}: {e}")
        data = None
    
    return data


# In[5]:


def read_pdf_file(file_path):
    """
    Reads a PDF file and returns the extracted text.
    
    Parameters:
        file_path (str): Path to the PDF file.
    
    Returns:
        text (str): Extracted text from the PDF.
    """
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"  # Extract text from each page
    except Exception as e:
        return f"An error occurred while reading the PDF file: {e}"

    return text.strip() if text else "No text could be extracted from the PDF."


# In[6]:


def ask_chatbot(question, file_data=None):
    prompt = f"Here is some data from a bioinformatics file:\n\n{file_data}\n\nBased on the above data, please answer the following question:\n{question}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a knowledgeable bioinformatics assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7,
        )

        answer = response.choices[0].message['content']
        return answer
    except Exception as e:
        return f"An error occurred while processing your request: {e}"


# In[7]:


def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for your question...")
        audio = recognizer.listen(source)

    try:
        question = recognizer.recognize_google(audio)
        print("You asked:", question)
    except sr.UnknownValueError:
        question = None
        print("Sorry, I could not understand your voice.")
    except sr.RequestError:
        question = None
        print("Could not request results; check your internet connection.")

    return question


# In[8]:


class SpeechEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.speaking_thread = None  # Store the speaking thread
        self.stop_speaking = False

    def speak(self, answer):
        """Speak out the answer using a text-to-speech engine."""
        self.engine.say(answer)
        self.engine.runAndWait()

    def speak_answer(self, answer):
        """Speak out the answer using a text-to-speech engine, with the ability to stop."""
        if self.speaking_thread and self.speaking_thread.is_alive():
            print("Already speaking. Please wait until the current speech is finished.")
            return

        self.stop_speaking = False
        self.speaking_thread = threading.Thread(target=self.speak, args=(answer,))
        self.speaking_thread.start()

        # Check for stop command while speaking
        while self.speaking_thread.is_alive():
            user_input = input("Enter 'stop' to stop speaking: ").strip().lower()
            if user_input == 'stop':
                self.engine.stop()
                self.stop_speaking = True
                print("Stopped speaking.")
                break

                


# In[9]:


def summarize_file_data(file_data):
    """
    Summarizes the given file data to reduce token count.
    
    Parameters:
        file_data (str): The content of the bioinformatics file.
    
    Returns:
        summary (str): A summarized version of the file data.
    """
    # Here we just return the first few lines as an example.
    # You can implement more sophisticated summarization logic based on the file format.
    lines = file_data.splitlines()
    relevant_lines = []  # To store relevant lines

    # Extract relevant information based on the file type (e.g., for VCF files)
    for line in lines:
        if line.startswith("#"):  # Skip header lines in VCF files
            continue
        relevant_lines.append(line)
        if len(relevant_lines) >= 10:  # Limit to first 10 relevant lines
            break

    return "\n".join(relevant_lines)  # Return summarized data


# In[10]:


def ask_chatbot(question, file_data=None):
    """
    Uses OpenAI's GPT model to answer the user's question based on file data and external knowledge.
    
    Parameters:
        question (str): The user's question.
        file_data (str): Content of the bioinformatics file, converted to a string.
    
    Returns:
        answer (str): The answer from the chatbot.
    """
    # Summarize the file data before using it in the prompt
    summarized_data = summarize_file_data(file_data)

    prompt = f"Based on the following summarized bioinformatics data:\n\n{summarized_data}\n\nPlease answer the following question:\n{question}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable bioinformatics assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,  # Limit the number of tokens for the answer as well
            temperature=0.5,
        )

        answer = response.choices[0].message['content']
        return answer
    except Exception as e:
        return f"An error occurred while processing your request: {e}"


# In[ ]:


# For PDF files
file_path = 'NS_wgs_report.pdf'  # Replace with your actual PDF file path
data = read_bioinformatics_file(file_path)

# Print the extracted content for verification
print("Extracted Content from PDF:")
print(data[:]) 


# In[1]:


#For Bioinformatics files

file_path = 'clinvar.vcf'
data = read_bioinformatics_file(file_path)

print("Extracted Content from the file:")
print(data[:]) 


# In[12]:


def ask_chatbot(question, file_data):
    """
    Sends a question to the OpenAI Chatbot and retrieves the answer.
    
    Parameters:
        question (str): The question to ask the chatbot.
        file_data (str): Additional data to provide context.
    
    Returns:
        answer (str): The chatbot's response.
    """
    try:
        # Limit context to avoid hitting the token limit
        context = file_data[:]  # Adjust this limit as necessary
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{context}\n\n{question}"}
            ],
            max_tokens=1500
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"An error occurred while processing your request: {e}"


# In[13]:


import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def ask_chatbot(question, file_data):
    try:
        context = file_data[:]  # Limit context length
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{context}\n\n{question}"}
            ],
            max_tokens=1500
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"Error during OpenAI API call: {e}")
        return "Sorry, I encountered an error while processing your request."


# In[2]:


continue_chat = True  # To control the loop for multiple questions
speech_engine = SpeechEngine()

while continue_chat:
    # Ask the user if they want to use text or voice input
    input_mode = input("Do you want to ask your question by typing (text) or speaking (voice)? Enter 'text' or 'voice': ").strip().lower()

    if input_mode == 'voice':
        question = get_voice_input()
    else:
        question = input("Please type your question: ")

    if question:
        # Ask the question using the chatbot logic
        answer = ask_chatbot(question, file_data=str(data))
        print("Chatbot Answer:", answer)
        
        # Ask the user if they want the answer spoken out loud
        output_mode = input("Do you want to hear the answer spoken aloud? Enter 'yes' or 'no': ").strip().lower()
        if output_mode == 'yes':
            speech_engine.speak_answer(answer)
        
        # Ask the user if they have more questions
        continue_response = input("Do you want to ask another question? Enter 'yes' or 'no': ").strip().lower()
        if continue_response != 'yes':
            continue_chat = False
            print("Thank you for using the chatbot. Goodbye!")


# In[ ]:




