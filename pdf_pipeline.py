import os  # Provides a way of using operating system-dependent functionality
import glob  # Finds all the pathnames matching a specified pattern
import logging  # Enables logging of events for tracking and debugging
from concurrent.futures import ThreadPoolExecutor, as_completed  # Facilitates concurrent execution of functions
from datetime import datetime  # Supplies classes for manipulating dates and times
import json  # Provides functions for working with JSON data

import PyPDF2  # Library for PDF file handling
from pymongo import MongoClient, errors  # MongoDB client and error handling
import nltk  # Natural Language Toolkit for text processing
from nltk.corpus import stopwords  # Provides a list of common stopwords
from nltk.tokenize import word_tokenize, sent_tokenize  # Functions to split text into words and sentences

from collections import defaultdict  # Dictionary subclass that calls a factory function to supply missing values
import math  # Provides access to mathematical functions

# Ensure NLTK data is downloaded
nltk.download('punkt')  # Downloads the Punkt tokenizer models
nltk.download('stopwords')  # Downloads the list of stopwords

# Configuration
PDF_FOLDER_PATH = '/Users/adityarounak/Desktop/AI-Internship-Task-Pipeline/pythonproject1/AI-Internship-Task-Pipeline/test'  # Path to the folder containing PDF files; replace with actual path
MONGODB_URI = 'mongodb://localhost:27017/'  # URI for connecting to MongoDB; replace with actual URI if different
DB_NAME = 'pdf_processing_db'  # Name of the MongoDB database to use
COLLECTION_NAME = 'pdf_documents'  # Name of the collection within the database
MAX_WORKERS = 4  # Number of threads to use for concurrent processing

# Logging Configuration
LOG_FILE = 'pdf_pipeline.log'  # Name of the log file to store logs
logging.basicConfig(
    level=logging.INFO,  # Sets the logging level to INFO
    format='%(asctime)s [%(levelname)s] %(message)s',  # Defines the log message format
    handlers=[
        logging.FileHandler(LOG_FILE),  # Logs messages to the specified log file
        logging.StreamHandler()  # Also logs messages to the console
    ]
)

# Initialize MongoDB Client
try:
    client = MongoClient(MONGODB_URI)  # Creates a MongoDB client using the specified URI
    db = client[DB_NAME]  # Accesses the specified database
    collection = db[COLLECTION_NAME]  # Accesses the specified collection within the database
    logging.info("Connected to MongoDB successfully.")  # Logs a successful connection message
except errors.ConnectionFailure as e:
    logging.error(f"Could not connect to MongoDB: {e}")  # Logs an error message if connection fails
    exit(1)  # Exits the script with an error code

# Define Stopwords
STOP_WORDS = set(stopwords.words('english'))  # Creates a set of English stopwords for quick lookup


def ingest_pdfs(folder_path):
    """
    Ingest all PDF files from the specified folder.

    Parameters:
        folder_path (str): The path to the folder containing PDF files.

    Returns:
        list: A list of PDF file paths.
    """
    pattern = os.path.join(folder_path, '*.pdf')  # Creates a pattern to match all PDF files in the folder
    pdf_files = glob.glob(pattern)  # Retrieves a list of all PDF files matching the pattern
    logging.info(f"Found {len(pdf_files)} PDF files in {folder_path}.")  # Logs the number of PDFs found
    return pdf_files  # Returns the list of PDF file paths


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.

    Parameters:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ""  # Initialize an empty string to hold the extracted text
    try:
        with open(pdf_path, 'rb') as file:  # Opens the PDF file in binary read mode
            reader = PyPDF2.PdfReader(file)  # Creates a PDF reader object
            for page_num in range(len(reader.pages)):  # Iterates over each page in the PDF
                page = reader.pages[page_num]  # Accesses the current page
                text += page.extract_text() or ""  # Extracts text from the page and appends to the text variable
        logging.info(f"Extracted text from {pdf_path}.")  # Logs successful text extraction
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")  # Logs any errors encountered
        raise e  # Raises the exception to be handled by the caller
    return text  # Returns the extracted text


def summarize_text(text, domain="general"):
    """
    Generate a domain-specific summary of the text.
    A simple implementation using sentence ranking based on word frequency.

    Parameters:
        text (str): The text to summarize.
        domain (str): The domain for which to generate the summary (unused in this implementation).

    Returns:
        str: The generated summary.
    """
    try:
        sentences = sent_tokenize(text)  # Splits the text into sentences
        if not sentences:
            return ""  # Returns an empty string if no sentences are found

        # Calculate word frequencies
        words = word_tokenize(text.lower())  # Tokenizes the text into words and converts to lowercase
        words = [word for word in words if
                 word.isalnum() and word not in STOP_WORDS]  # Filters out non-alphanumeric words and stopwords
        freq = defaultdict(int)  # Initializes a default dictionary for word frequencies
        for word in words:
            freq[word] += 1  # Counts the frequency of each word

        # Calculate sentence scores
        sentence_scores = defaultdict(float)  # Initializes a default dictionary for sentence scores
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):  # Tokenizes each sentence into words
                if word in freq:
                    sentence_scores[sentence] += freq[word]  # Adds the word frequency to the sentence score

        # Select top N sentences for summary
        summary_length = determine_summary_length(len(sentences))  # Determines the number of sentences for the summary
        summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[
                            :summary_length]  # Selects top scoring sentences
        summary = ' '.join(summary_sentences)  # Joins the selected sentences into a single summary string
        logging.info("Summary generated.")  # Logs successful summary generation
    except Exception as e:
        logging.error(f"Error during summarization: {e}")  # Logs any errors encountered during summarization
        summary = ""  # Sets summary to an empty string in case of error
    return summary  # Returns the generated summary


def determine_summary_length(num_sentences):
    """
    Determine the number of sentences in the summary based on document length.

    Parameters:
        num_sentences (int): The total number of sentences in the document.

    Returns:
        int: The number of sentences to include in the summary.
    """
    if num_sentences <= 10:
        return max(1, num_sentences // 2)  # For short documents, use half the sentences, at least one
    elif num_sentences <= 30:
        return max(2, num_sentences // 3)  # For medium documents, use a third of the sentences, at least two
    else:
        return max(3, num_sentences // 4)  # For long documents, use a quarter of the sentences, at least three


def extract_keywords(text, domain="general"):
    """
    Extract domain-specific keywords using Term Frequency (TF).

    Parameters:
        text (str): The text from which to extract keywords.
        domain (str): The domain for which to extract keywords (unused in this implementation).

    Returns:
        list: A list of extracted keywords.
    """
    try:
        words = word_tokenize(text.lower())  # Tokenizes the text into words and converts to lowercase
        words = [word for word in words if
                 word.isalnum() and word not in STOP_WORDS]  # Filters out non-alphanumeric words and stopwords

        # Calculate term frequencies
        tf = defaultdict(int)  # Initializes a default dictionary for term frequencies
        for word in words:
            tf[word] += 1  # Counts the frequency of each word

        # Since we're processing documents individually, IDF can be approximated or set to 1
        # For simplicity, we'll use TF as the score
        sorted_tf = sorted(tf.items(), key=lambda item: item[1],
                           reverse=True)  # Sorts words by frequency in descending order
        top_n = 10  # Number of top keywords to extract
        keywords = [word for word, freq in sorted_tf[:top_n]]  # Extracts the top N keywords
        logging.info("Keywords extracted.")  # Logs successful keyword extraction
    except Exception as e:
        logging.error(f"Error during keyword extraction: {e}")  # Logs any errors encountered during keyword extraction
        keywords = []  # Sets keywords to an empty list in case of error
    return keywords  # Returns the list of extracted keywords


def process_pdf(pdf_path):
    """
    Process a single PDF: extract text, summarize, extract keywords, and update MongoDB.

    Parameters:
        pdf_path (str): The path to the PDF file.
    """
    try:
        # Extract metadata
        document_name = os.path.basename(pdf_path)  # Extracts the file name from the path
        file_size = os.path.getsize(pdf_path)  # Gets the size of the PDF file in bytes
        file_size_kb = f"{file_size / 1024:.2f} KB"  # Converts file size to kilobytes with two decimal places
        document_path = os.path.abspath(pdf_path)  # Gets the absolute path of the PDF file
        metadata = {
            "document_name": document_name,  # Name of the document
            "path": document_path,  # Path to the document
            "size": file_size_kb,  # Size of the document
            "ingested_at": datetime.utcnow()  # Timestamp of ingestion in UTC
        }

        # Insert initial metadata into MongoDB
        result = collection.insert_one(metadata)  # Inserts the metadata document into MongoDB
        document_id = result.inserted_id  # Retrieves the unique ID of the inserted document
        logging.info(f"Inserted metadata for {document_name} with ID {document_id}.")  # Logs successful insertion

        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)  # Calls the function to extract text from the PDF
        if not text.strip():
            raise ValueError("No text extracted from PDF.")  # Raises an error if no text is extracted

        # Generate summary
        summary = summarize_text(text)  # Calls the function to generate a summary of the text

        # Extract keywords
        keywords = extract_keywords(text)  # Calls the function to extract keywords from the text

        # Prepare JSON update
        update_data = {
            "summary": summary,  # Adds the generated summary
            "keywords": keywords,  # Adds the extracted keywords
            "processed_at": datetime.utcnow()  # Timestamp of processing completion in UTC
        }

        # Update MongoDB document
        collection.update_one({"_id": document_id},
                              {"$set": update_data})  # Updates the MongoDB document with summary and keywords
        logging.info(
            f"Updated MongoDB document for {document_name} with summary and keywords.")  # Logs successful update

    except Exception as e:
        logging.error(f"Failed to process {pdf_path}: {e}")  # Logs any errors encountered during processing
        # Optionally, update MongoDB with error info
        collection.update_one(
            {"document_name": os.path.basename(pdf_path)},  # Filters the document by document name
            {"$set": {"error": str(e), "processed_at": datetime.utcnow()}}
            # Sets the error message and processing timestamp
        )


def main():
    """
    Main function to execute the PDF processing pipeline.
    """
    pdf_files = ingest_pdfs(PDF_FOLDER_PATH)  # Calls the function to ingest all PDFs from the specified folder
    if not pdf_files:
        logging.info("No PDF files to process.")  # Logs if no PDFs are found
        return  # Exits the function if there are no PDFs to process

    with ThreadPoolExecutor(
            max_workers=MAX_WORKERS) as executor:  # Creates a thread pool executor for concurrent processing
        future_to_pdf = {executor.submit(process_pdf, pdf): pdf for pdf in
                         pdf_files}  # Submits all PDF processing tasks to the executor
        for future in as_completed(future_to_pdf):  # Iterates over the completed futures as they finish
            pdf = future_to_pdf[future]  # Retrieves the PDF path associated with the completed future
            try:
                future.result()  # Retrieves the result of the future, raising any exceptions if occurred
                logging.info(f"Processing completed for {pdf}.")  # Logs successful processing of the PDF
            except Exception as e:
                logging.error(f"Error processing {pdf}: {e}")  # Logs any errors encountered during processing

    logging.info("All PDF files have been processed.")  # Logs completion of processing all PDFs


if __name__ == "__main__":
    main()  # Calls the main function to start the pipeline
