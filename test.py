import pandas as pd
import fecir

NUM_DATA = 1000
NUM_CLUSTER = 10

# Selecting list of documents to be read
#input_file = "./data/template.csv"      # must be replaced with a file containing the corpus
input_file = "./data/proposicao-tema-completo.csv"      # must be replaced with a file containing the corpus
df = pd.read_csv(input_file, delimiter=',', encoding="utf-8")
#data = df.corpus        # replace with the desired column name
data = df.txtEmenta.dropna()        # replace with the desired column name
data = data[:NUM_DATA].to_numpy()

# Running preprocessing step
model = fecir.Retriever(data)
model.build(k=NUM_CLUSTER, verbose=True)


#query = "Example query for the information retrieval step."

query = "Solicito reelabora��o do PL 10235/2018, para reapresenta��o. Sugest�es de adequa��o t�cnica e legislativas para viabilizar que a mat�ria prospere ser�o muito bem vindas.  Ementa:  Altera a Lei n� 7.827, de 27 de setembro de 1989, para incluir os munic�pios do norte de Goi�s na �rea de aplica��o de recursos do Fundo Constitucional de Financiamento do Norte."
docs = model.search(query, cluster_select="highest", doc_select="full")


for d in docs:
    print()
    print(d)
    print()
