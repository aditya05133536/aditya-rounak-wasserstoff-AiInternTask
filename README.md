# aditya-rounak-wasserstoff-AiInternTask
# AI Internship Task - PDF Processing Pipeline

## Overview

This repository contains a custom-built pipeline to process PDF documents, generate domain-specific summaries and keywords, and store the results in a MongoDB database.

## Features

- **PDF Ingestion:** Monitors a specified folder for new PDF files and ingests them automatically.
- **Concurrent Processing:** Utilizes multi-processing to handle multiple PDFs in parallel.
- **Summarization:** Generates concise summaries of the extracted text based on document length.
- **Keyword Extraction:** Identifies relevant, non-generic keywords from the text.
- **MongoDB Integration:** Stores and updates processed data in a MongoDB collection.
- **Resource Management:** Monitors system CPU and memory usage to prevent overloading.
- **Comprehensive Logging:** Logs all operations, successes, and errors for easy monitoring and troubleshooting.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/AI-Internship-Task.git
cd AI-Internship-Task
