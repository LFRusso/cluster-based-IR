from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import fecir

#NUM_DATA = 1000


input_file = "./data/proposicao-tema-completo-sem-duplicado-2019.csv"
df = pd.read_csv(input_file, delimiter=',', encoding='utf-8')

data = df.txtEmenta.dropna().to_numpy()
#data = data[:NUM_DATA]

model = fecir.Retriever(data)
for i in range(3, 20):
    model.build(k=i)
    cluster_labels = model.doc_labels
    X = model.vsm

    silhouette_avg = silhouette_score(X, cluster_labels)    
    print("For n_clusters =", i,
          "The average silhouette_score is :", silhouette_avg)
