import pandas as pd
import fecir

NUM_DATA = 100
NUM_CLUSTER = 10

# Selecting list of documents to be read
input_file = "./data/template.csv"      # must be replaced with a file containing the corpus
df = pd.read_csv(input_file, delimiter=',', encoding="utf-8")
data = df.corpus        # replace with the desired column name
data = data[:NUM_DATA] 

# Running preprocessing step
model = fecir.Retriever(data)
model.build(k=NUM_CLUSTER)


query = "Example query for the information retrieval step."
ranked_clusters = model.search(query)