import os
import pickle

class DocumentProcessor:
    """
    Suddivide il testo in chunk e salva/carica la cache dei chunk.
    """
    def __init__(self, chars_per_chunk: int = 5000, cache_path: str = 'chunks.pkl'):
        self.chars_per_chunk = chars_per_chunk
        self.cache_path = cache_path

    def load_or_process(self, documents: dict) -> list[str]:
        # 1) Prova a caricare cache
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass  # se fallisce, ricalcola

        # 2) Calcola chunk
        chunks = []
        for text in documents.values():
            if not text:
                continue
            for i in range(0, len(text), self.chars_per_chunk):
                chunks.append(text[i:i+self.chars_per_chunk])

        # 3) Salva cache
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(chunks, f)
        except Exception:
            pass
        return chunks