import pandas as pd
import json

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    return pd.DataFrame(data)

def remove_duplicates(data):
    return data.drop_duplicates()