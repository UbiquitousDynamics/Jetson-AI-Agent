import logging
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer
import config
import threading

class APIClient:
    def __init__(self, 
                 model_talk: str = config.MODEL_TALK, 
                 model_think: str = config.MODEL_THINK, 
                 device: int = -1,  # -1 per CPU, 0 per la prima GPU, 1 per la seconda, ecc.
                 cache_dir: str = "models_cache"):
        """
        Inizializza i modelli locali di Hugging Face.
        Se il modello non è presente nella cache, verrà scaricato automaticamente.
        Impostando device=-1 si utilizza la CPU; impostando device=0 si utilizza la prima GPU.
        """
        os.makedirs(cache_dir, exist_ok=True)
        logging.debug("Caricamento dei modelli locali di Hugging Face...")
        self.talk_pipeline = self._load_model(model_talk, device, cache_dir, getattr(config, "MODEL_TALK_REVISION", None))
        self.think_pipeline = self._load_model(model_think, device, cache_dir, getattr(config, "MODEL_THINK_REVISION", None))
    
    def _load_model(self, model_name: str, device: int, cache_dir: str, revision: str = None):
        logging.info(f"Caricamento del modello {model_name} dalla cache: {cache_dir}")
        # Carica il modello e il tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, revision=revision)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, revision=revision)
        # Sposta il modello sul dispositivo specificato
        model.to(torch.device(f"cuda:{device}" if device >= 0 else "cpu"))
        logging.debug(f"Modello {model_name} caricato con successo.")
        # Stampa le informazioni sul devicie
        logging.debug(f"Dispositivo utilizzato: {torch.cuda.get_device_name(device) if device >= 0 else 'CPU'}")
        # Crea la pipeline specificando il dispositivo
        return pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    
    def talk(self, message: str, context: str = None) -> str:
        """
        Genera una risposta per il prompt 'message', aggiungendo eventualmente il contesto.
        """
        prompt = f"Context:\n{context}\n\nPrompt:\n{message}" if context else message
        logging.debug(f"Generazione del testo con il modello talk, prompt: {prompt[:50]}...")
        outputs = self.talk_pipeline(prompt, max_new_tokens=50, truncation=True)
        generated_text = outputs[0]["generated_text"]
        logging.debug(f"Testo generato: {generated_text[:50]}...")
        return generated_text

    def talk_stream(self, message: str, context: str = None) -> str:
        """
        Genera una risposta in streaming, processando l'output token per token.
        """
        prompt = f"Context:\n{context}\n\nPrompt:\n{message}" if context else message
        logging.debug(f"Generazione del testo con il modello talk, prompt: {prompt[:50]}...")
    
        # Prepara gli input per il modello (abilita truncation qui se necessario)
        inputs = self.talk_pipeline.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.talk_pipeline.model.device) for k, v in inputs.items()}
    
        # Configura lo streamer per ricevere token in tempo reale
        streamer = TextIteratorStreamer(self.talk_pipeline.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
        # Parametri di generazione senza il parametro 'truncation'
        generation_kwargs = dict(
            max_new_tokens=50,
            streamer=streamer
        )

        thread = threading.Thread(target=self.talk_pipeline.model.generate, kwargs={**inputs, **generation_kwargs})
        thread.start()
    
        generated_text = ""
        # Itera sui token man mano che vengono generati
        for token in streamer:
            generated_text += token
            # Processa il token (es. stampalo subito)
            print(token, end="", flush=True)
    
        logging.debug(f"Testo generato: {generated_text[:50]}...")
        return generated_text
    
    def talk_stream_tts(self, message: str, context: str = None, tts_callback=None) -> None:
        """
        Genera una risposta in streaming e la passa al callback TTS parola per parola.
        Se tts_callback è None, stampa semplicemente l'output.
        """
        prompt = f"Context:\n{context}\n\nPrompt:\n{message}" if context else message
        logging.debug(f"Generazione del testo con il modello talk, prompt: {prompt[:50]}...")

        # Prepara gli input per il modello (abilita truncation se necessario)
        inputs = self.talk_pipeline.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.talk_pipeline.model.device) for k, v in inputs.items()}

        # Configura lo streamer per ricevere token in tempo reale
        streamer = TextIteratorStreamer(self.talk_pipeline.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            max_new_tokens=50,
            streamer=streamer
        )

        # Avvia la generazione in un thread separato
        thread = threading.Thread(target=self.talk_pipeline.model.generate, kwargs={**inputs, **generation_kwargs})
        thread.start()

        buffer = ""  # Buffer per accumulare il testo incompleto
        # Itera sui token man mano che vengono generati
        for token in streamer:
            print(token, end="", flush=True)
            buffer += token

            # Controlla che il buffer non sia vuoto prima di usarlo
            if not buffer:
                continue

            # Suddividi il buffer in parole usando lo spazio come delimitatore
            words = buffer.split(" ")

            # Se l'ultimo carattere del buffer non è uno spazio, l'ultima parola è incompleta
            if buffer and buffer[-1] != " ":
                complete_words = words[:-1]
            else:
                complete_words = words

            for word in complete_words:
                if word.strip():
                    if tts_callback:
                        tts_callback(word)
                    else:
                        print(f"\n[TTS] {word}")
            # Aggiorna il buffer con l'ultima parola (potrebbe essere incompleta)
            buffer = words[-1] if words else ""

        # Dopo lo streaming, se rimane una parola incompleta, inviala al TTS
        if buffer.strip():
            if tts_callback:
                tts_callback(buffer.strip())
            else:
                print(f"\n[TTS] {buffer.strip()}")
                
    def talk_stream_tts_phrase(self, message: str, context: str = None, tts_callback=None) -> None:
        """
        Genera una risposta in streaming e la passa al callback TTS frase per frase.
        Se tts_callback è None, stampa semplicemente l'output.
        """
        prompt = f"Context:\n{context}\n\nPrompt:\n{message}" if context else message
        logging.debug(f"Generazione del testo con il modello talk, prompt: {prompt[:50]}...")

        # Prepara gli input per il modello (abilita truncation se necessario)
        inputs = self.talk_pipeline.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.talk_pipeline.model.device) for k, v in inputs.items()}

        # Configura lo streamer per ricevere token in tempo reale
        streamer = TextIteratorStreamer(self.talk_pipeline.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            max_new_tokens=50,
            streamer=streamer
        )

        # Avvia la generazione in un thread separato
        thread = threading.Thread(target=self.talk_pipeline.model.generate, kwargs={**inputs, **generation_kwargs})
        thread.start()

        sentence_buffer = ""
        # Itera sui token man mano che vengono generati
        for token in streamer:
            print(token, end="", flush=True)
            sentence_buffer += token

            # Se il buffer non è vuoto e termina con un segno di fine frase, invia la frase al TTS
            if sentence_buffer.strip() and sentence_buffer.strip()[-1] in {'.', '!', '?'}:
                if tts_callback:
                    tts_callback(sentence_buffer.strip())
                else:
                    print(f"\n[TTS] {sentence_buffer.strip()}")
                sentence_buffer = ""  # Resetta il buffer

        # Se alla fine rimane una frase incompleta, inviala comunque
        if sentence_buffer.strip():
            if tts_callback:
                tts_callback(sentence_buffer.strip())
            else:
                print(f"\n[TTS] {sentence_buffer.strip()}")
    
    def think(self, message: str, context: str = None) -> str:
        """
        Genera una risposta per il prompt 'message' utilizzando il modello think, con contesto opzionale.
        """
        prompt = f"Context:\n{context}\n\nPrompt:\n{message}" if context else message
        logging.debug(f"Generazione del testo con il modello think, prompt: {prompt[:50]}...")
        outputs = self.think_pipeline(prompt, max_new_tokens=50, truncation=True)
        generated_text = outputs[0]["generated_text"]
        logging.debug(f"Testo generato: {generated_text[:50]}...")
        return generated_text
