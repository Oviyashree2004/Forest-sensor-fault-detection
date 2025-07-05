def preprocess_data(df):
    df = df.dropna()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df
