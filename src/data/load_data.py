import pandas as pd

def load_data(file_path="flight_data.csv"):
    """
    Load the cleaned dataset and drop unnecessary columns.
    """
    try:
        data = pd.read_csv(file_path)
        # Drop the unnamed index column if it exists
        if "Unnamed: 0" in data.columns:
            data.drop(columns="Unnamed: 0", inplace=True)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None