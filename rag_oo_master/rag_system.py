import os
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


class RagSystem:
    def __init__(self, model_name='BAAI/bge-m3', use_fp16=True):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.embedding_matrix = None
        self.data = None  # Original sentences

    def index_database(self, data, save_path='embeddings.npy', text_data_path='sentences.txt'):
        """
        Encode the input data and save the embeddings only if they don't already exist.
        Also saves the original sentences to compare on reload.
        """
        self.data = data

        # Controlla se gli embeddings esistono già
        if os.path.exists(save_path) and os.path.exists(text_data_path):
            with open(text_data_path, 'r', encoding='utf-8') as f:
                saved_sentences = [line.strip() for line in f.readlines()]

            # Se le frasi sono le stesse, ricarica gli embeddings
            if saved_sentences == data:
                print("Found existing embeddings. Loading them from disk.")
                self.embedding_matrix = np.load(save_path)
                return

        print("Computing new embeddings and saving them.")
        embeddings = self.model.encode(data)['dense_vecs']
        self.embedding_matrix = embeddings

        np.save(save_path, embeddings)

        # Salva anche le frasi per il confronto futuro
        with open(text_data_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(data))

    def load_embedding_matrix(self, embeddings_path):
        self.embedding_matrix = np.load(embeddings_path)
        return self.embedding_matrix

    def search(self, query, top_k=None):
        if self.embedding_matrix is None:
            raise ValueError("Embedding matrix not loaded. Run index_database() or load_embedding_matrix() first.")

        query_embedding = self.model.encode([query])['dense_vecs']
        similarities = cosine_similarity(query_embedding, self.embedding_matrix)[0]
        similarities_results = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

        print("\nSimilarity with the query (in descending order):\n")
        for idx, similarity in (similarities_results if not top_k else similarities_results[:top_k]):
            print(f"Index: {idx} | Similarity: {similarity:.4f}")

        return similarities_results

    def visualize_space_query(self, query, perplexity=2):
        if self.embedding_matrix is None or self.data is None:
            raise ValueError("You must first run index_database() to have both data and embeddings.")

        query_embedding = self.model.encode([query])['dense_vecs']
        jointed_matrix = np.vstack((self.embedding_matrix, query_embedding))

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(jointed_matrix)

        plt.figure(figsize=(8, 6))
        plt.scatter(embeddings_2d[:-1, 0], embeddings_2d[:-1, 1], color='blue', edgecolors='k', label='Sentences')
        plt.scatter(embeddings_2d[-1, 0], embeddings_2d[-1, 1], color='red', edgecolors='k', label='Query')

        for i, sentence in enumerate(self.data):
            plt.text(embeddings_2d[i, 0] + 0.5, embeddings_2d[i, 1], sentence, fontsize=8)

        plt.text(embeddings_2d[-1, 0] + 0.5, embeddings_2d[-1, 1], f"Query: {query}", fontsize=9, color='red')
        plt.title("t-SNE Visualization of Sentence Embeddings with Query")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
