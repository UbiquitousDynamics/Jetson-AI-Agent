
# Configure ARCHITECTURE
ARCHITECTURE= "aarch64"

# Configure OS
OS = "windows"

# Configuration constants for the speech synthesizer
LANGUAGE = "en"
TTS_FOLDER = 'tts_audio'
TTS = "tts"

# Configuration constants for document loading
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# Configuration constants for STT vosk model
VOSK_MODEL_PATH = "None"

# Configuration constants for the API client
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_TALK = "gpt2"
MODEL_THINK = "gpt2"

# Configuration constants for the speech recognizer
WAKE_WORD = 'hello'
LISTEN_TIMEOUT = 10
WAKE_SOUND = 'sounds/wake_up.mp3'
STOP_SOUND = 'sounds/stop.mp3'
TIMEOUT_SOUND = 'sounds/stop.mp3'

# Configuration for sound player
SOUND_PLAYER = "os_sound_player"

if ARCHITECTURE == "aarch64":
   WAKE_SOUND = 'sounds/wake_up.wav'
   STOP_SOUND = 'sounds/stop.wav'
   TIMEOUT_SOUND = 'sounds/stop.wav'

# switch language
if LANGUAGE == "en": 
    VOSK_MODEL_PATH = "recognizer/models/vosk-model-small-en-us-0.15"
    VOICE = "mb-us1" # Select MBROLA-voices, example: mb-us1 (american english female voice); mb-it4 (italian female voice)
elif LANGUAGE == "it":
    VOSK_MODEL_PATH = "recognizer/models/vosk-model-small-it-0.22"
    VOICE = "mb-it4" # Select MBROLA-voices, example: mb-us1 (american english female voice); mb-it4 (italian female voice) 


#switch OS imports
if OS == "windows":
   SOUND_PLAYER = "sound_player"
   TTS = "win_tts"
