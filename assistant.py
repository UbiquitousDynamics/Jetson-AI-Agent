import logging
import time
from threading import Thread, Lock, Event
import queue
import speech_recognition as sr
from api.api_client_advanced import APIClient
from recognizer.speech_recognizer import SpeechRecognizer
# from document.document_retriever import DocumentRetriever
from document.rag_document_retriver import RagDocumentRetriever
import speech_recognition as sr
import config

if config.OS_ == config.OS.WINDOWS:
    from audio.win_tts import Pyttsx3TTS
    from audio.sound_player import SoundPlayer
else:
    from audio.tts import Pyttsx3TTS
    from audio.os_sound_player import SoundPlayer

class VoiceAssistant:
    def __init__(self, documents: dict = None):
        self.api = APIClient()
        self.tts = Pyttsx3TTS()
        self.sound = SoundPlayer()
        self.rec_main = SpeechRecognizer()
        self.rec_bg = SpeechRecognizer()
        self.doc_retriever = RagDocumentRetriever(documents) if documents else None

        self.interrupt_event = Event()
        self.shutdown_event = Event()
        self.sound_queue = queue.Queue()
        self.audio_lock = Lock()
        
        # Aggiunge storia della conversazione
        self.conversation_history = []
        
        # Protegge contro riconoscimenti duplicati
        self.last_command = ""
        self.last_command_time = 0
        self.command_cooldown = 1.5  # secondi

        # Thread per suoni
        self.sound_thread = Thread(target=self._sound_worker, daemon=True)
        self.sound_thread.start()

    def _sound_worker(self):
        while not self.shutdown_event.is_set():
            try:
                snd = self.sound_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self.sound.play_sound(snd)
            self.sound_queue.task_done()
        logging.info("Sound worker terminato")

    def play_sound(self, sound_id):
        if not self.shutdown_event.is_set():
            self.sound_queue.put_nowait(sound_id)

    def listen_wake(self):
        for _ in range(3):
            if self.shutdown_event.is_set():
                return False
            try:
                return self.rec_main.listen_for_wake_word(config.WAKE_WORD)
            except Exception as e:
                logging.error(f"Wake-listen error: {e}")
                time.sleep(0.1)
        return False

    def is_duplicate_command(self, command):
        """Verifica se un comando è un duplicato recente."""
        current_time = time.time()
        if (command == self.last_command and 
            current_time - self.last_command_time < self.command_cooldown):
            return True
        
        self.last_command = command
        self.last_command_time = current_time
        return False

    def listen_command(self, timeout=None):
        try:
            with self.audio_lock:
                command = self.rec_main.listen(timeout=timeout)
                
                # Verifica se è un duplicato
                if command and self.is_duplicate_command(command):
                    logging.debug(f"Comando duplicato ignorato: {command}")
                    return ""
                    
                return command
        except sr.WaitTimeoutError:
            return ""
        except Exception as e:
            logging.error(f"Listen error: {e}")
            return ""

    def process(self, command: str):
        if not command:
            return
            
        self.interrupt_event.clear()
        
        # Aggiunge il comando alla storia
        self.conversation_history.append({"role": "user", "content": command})
        
        # Limita la storia a 5 scambi per non sovraccaricare il contesto
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
            
        # Formatta la storia come contesto
        conversation_context = "\n".join([
            f"{'User' if item['role'] == 'user' else 'Assistant'}: {item['content']}"
            for item in self.conversation_history
        ])
        
        context = ""
        if self.doc_retriever:
            try:
                docs = self.doc_retriever.retrieve(command)
                docs_context = "\n".join(d for _, d in docs)
                if docs_context:
                    context = f"Relevant documents:\n{docs_context}\n\nConversation history:\n{conversation_context}"
                else:
                    context = f"Conversation history:\n{conversation_context}"
            except Exception as e:
                logging.error(f"Doc retrieve error: {e}")
                context = f"Conversation history:\n{conversation_context}"
        else:
            context = f"Conversation history:\n{conversation_context}"

        # Crea un prompt appropriato in base al tipo di comando
        if "think" in command.lower():
            logging.info(f"Elaborando comando 'think': {command}")
            resp = self.api.think(command, context)
            self._speak_interruptible(resp)
            # Aggiungi la risposta alla storia
            self.conversation_history.append({"role": "assistant", "content": resp})
        else:
            # Per comandi normali, aggiungiamo istruzioni specifiche
            enhanced_context = context
            if "come stai" in command.lower():
                enhanced_context = f"{context}\n\nIstruzioni: L'utente ha chiesto 'come stai'. Rispondi con uno stato d'animo e non con una definizione di parole."
                
            logging.info(f"Elaborando comando: {command}")
            
            # Cattura la risposta per aggiungerla alla storia
            response_buffer = []
            
            def capture_response(chunk):
                if self.interrupt_event.is_set():
                    return False
                response_buffer.append(chunk)
                self.tts.speak(chunk)
                return True
                
            self.api.talk_stream_tts_phrase(
                command,
                enhanced_context,
                tts_callback=capture_response,
                command_callback=lambda c: self.interrupt_event.set(),
            )
            
            # Aggiungi la risposta completa alla storia
            full_response = " ".join(response_buffer)
            if full_response:
                self.conversation_history.append({"role": "assistant", "content": full_response})

    def _tts_callback(self, chunk):
        if self.interrupt_event.is_set():
            return False
        self.tts.speak(chunk)
        return True

    def _speak_interruptible(self, text):
        # Speaks in chunk di frasi
        for sent in text.split(". "):
            if self.interrupt_event.is_set():
                break
            self.tts.speak(sent.strip())

    def run(self):
        logging.info("Assistant avviato, in attesa di wake word")
        while not self.shutdown_event.is_set():
            if self.listen_wake():
                logging.info("Wake word detected!")
                self.play_sound(config.WAKE_SOUND)
                
                # Cooldown dopo il rilevamento del wake word
                time.sleep(0.5)
                
                while not self.shutdown_event.is_set():
                    logging.info("Listening...")
                    cmd = self.listen_command(config.LISTEN_TIMEOUT)
                    
                    if not cmd:
                        logging.info("Timeout o comando vuoto")
                        self.play_sound(config.TIMEOUT_SOUND)
                        break
                        
                    if any(k in cmd.lower() for k in ("stop", "exit", "quit", "termina")):
                        logging.info(f"Comando di uscita ricevuto: {cmd}")
                        self.play_sound(config.STOP_SOUND)
                        break
                        
                    logging.info(f"Processando comando: '{cmd}'")
                    self.process(cmd)
                    
                    # Aggiungi un cooldown dopo l'elaborazione
                    time.sleep(1.0)
                    
            time.sleep(0.1)
        self.shutdown()

    def shutdown(self):
        logging.info("Shutting down Assistant")
        self.shutdown_event.set()
        self.sound_thread.join(timeout=1.0)