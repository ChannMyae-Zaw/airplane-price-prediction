import pandas as pd

def build_features(data):
    """
    Prepare features for the ML pipeline.
    Currently drops unnecessary columns.
    """
    data = data.copy()
    
    # Drop unneeded columns
    if "flight" in data.columns:
        data.drop(columns=["flight"], inplace=True)
    
    return data