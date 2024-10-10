import pandas as pd
import json


def json_to_csv(json_file_path, csv_file_path):
    """
    Converts a JSON file to CSV format.

    Parameters:
        json_file_path (str): The path to the input JSON file.
        csv_file_path (str): The path where the output CSV will be saved.
    """
    try:
        # Load JSON data
        with open("/Users/adityarounak/Desktop/AI-Internship-Task-Pipeline/pythonproject1/AI-Internship-Task-Pipeline/pdf_pipeline.documents.json", 'r', encoding='utf-8') as file:
            data = json.load(file)

        # If the JSON data is a list of dictionaries
        if isinstance(data, list):
            # Normalize semi-structured JSON data into a flat table
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            # If JSON is a single dictionary, convert it into a DataFrame
            df = pd.json_normalize([data])
        else:
            raise ValueError("Unsupported JSON structure")

        # Save DataFrame to CSV
        df.to_csv("/Users/adityarounak/Desktop/AI-Internship-Task-Pipeline/pythonproject1/AI-Internship-Task-Pipeline/pdf.csv", index=False)
        print(f"Successfully converted {json_file_path} to {csv_file_path}")

    except Exception as e:
        print(f"Error converting JSON to CSV: {e}")


if __name__ == "__main__":
    # Replace these paths with your actual file paths
    input_json = 'input.json'
    output_csv = 'output.csv'

    json_to_csv(input_json, output_csv)
