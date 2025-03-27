import pygroupf

processor = pygroupf.data_processing.DataProcessor("german_credit_data.csv")
df = processor.clean_data()
print(df.head())
