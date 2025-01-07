from openai import OpenAI
import pandas as pd
import json
import streamlit as st
import csv
from io import StringIO

class SyntheticDataGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)


    def generate_tabular_data(self, reference_data: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        # Adjust the prompt to request CSV-format output from the model
        prompt = f"""
                    Generate {num_rows} rows of synthetic data in CSV format based on the following schema:
                    {reference_data.head().to_string(index=False)}

                    Please provide the output in the following CSV format:

                    START_CSV
                    ID,Name,Age,Department,Salary,Date_of_Joining
                    6,Anna Lang,32,Marketing,68000,2020-01-10
                    7,Liam Jones,35,Engineering,75000,2016-05-07
                    8,Julia Clark,27,HR,62000,2021-03-15
                    9,Ethan Lopez,38,Engineering,81000,2018-08-25
                    10,Olivia Garcia,34,Marketing,69000,2019-09-02
                    END_CSV
                """
        

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data generation assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        synthetic_data_text = response.choices[0].message.content
        st.write(synthetic_data_text)

        # Extract the CSV content between START_CSV and END_CSV
        start_index = synthetic_data_text.find("START_CSV") + len("START_CSV")
        end_index = synthetic_data_text.find("END_CSV")
        csv_data = synthetic_data_text[start_index:end_index].strip()

        # Clean up the CSV data if needed (optional)
        csv_data = "\n".join([line.strip() for line in csv_data.split("\n") if line.strip()])
        # Replace double spaces with commas (if needed for proper CSV formatting)
        csv_data = csv_data.replace('  ', ',')
        # Parse the CSV data into a DataFrame
        synthetic_data = pd.read_csv(StringIO(csv_data))
        return synthetic_data
    


    def generate_textual_data(self, reference_text: str, num_samples: int) -> list:
        prompt = f"Generate {num_samples} synthetic samples based on the following text:\n{reference_text}"
        response = self.client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a text generation assistant."},
            {"role": "user", "content": prompt}
        ])
        synthetic_texts = response.choices[0].message.content.split("\n")
        return synthetic_texts
    
    

    def generate_textual_data_with_schema(self, reference_data: pd.DataFrame, text_column: str, num_samples: int) -> pd.DataFrame:
        """
        Generate synthetic textual data while preserving the schema of the reference dataset.
        """
        import csv
        from io import StringIO
        
        # Extract a few sample rows for context
        sample_rows = reference_data.head(5).to_dict(orient='records')
        schema = list(reference_data.columns)
        
        # Build the prompt
        prompt = f"""
        Generate {num_samples} rows of synthetic data strictly following the schema:
        Columns: {schema}
        
        - Focus on generating synthetic text for the column: '{text_column}'.
        - For other columns ('Category' and 'Product'), provide contextually relevant synthetic values.
        - Ensure the output matches the schema exactly: {schema}
        - The output must be in CSV format with the header row: {','.join(schema)}
        
        Here are some example rows from the dataset:
        {json.dumps(sample_rows, indent=2)}
        
        Example Output:
        {','.join(schema)}
        Synthetic incident text 1,Category Example,Product Example
        Synthetic incident text 2,Another Category,Another Product
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant generating synthetic data that strictly follows a given schema."},
                {"role": "user", "content": prompt}
            ]
        )

        # Parse the response
        response_text = response.choices[0].message.content.strip()
        
        # Check if the response starts with the correct schema
        reader = csv.reader(StringIO(response_text))
        headers = next(reader)
        headers = [h.strip() for h in headers]
        
        if headers != schema:
            raise ValueError(
                f"Schema mismatch!\nExpected: {schema}\nGot: {headers}\nResponse:\n{response_text}"
            )
        
        # Load the CSV data into a DataFrame
        synthetic_data = pd.DataFrame(reader, columns=headers)
        return synthetic_data






        

        
