
from enum import Enum
import platform

class Architecture(Enum):
    X86 = "x86"
    AARCH64 = "aarch64"

class OS(Enum):
    WINDOWS = "windows"
    LINUX = "linux"

class Language(Enum):
    ITALIAN = "it"
    ENGLISH = "en"

class Device(Enum):
    CPU = -1
    GPU = 0 # Assuming GPU index 0 for simplicity
    GPU_1 = 1
    GPU_2 = 2

DEVICE = Device.GPU  # Set the default device to CPU
LANGUAGE = Language.ENGLISH  # Set the default language

# Detect the OS and set the default OS
detected_os = platform.system().lower()
if "windows" in detected_os:
    OS_ = OS.WINDOWS
elif "linux" in detected_os:
    OS_ = OS.LINUX
else:
    raise ValueError(f"Unsupported operating system: {detected_os}")

# Detect the architecture and set the default architecture
detected_arch = platform.machine().lower()
if "x86" in detected_arch or "amd64" in detected_arch:
    ARCHITECTURE = Architecture.X86
elif "aarch64" in detected_arch or "arm64" in detected_arch:
    ARCHITECTURE = Architecture.AARCH64
else:
    raise ValueError(f"Unsupported architecture: {detected_arch}")

TTS_FOLDER = 'tts_audio'

# Configuration constants for document loading
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# Configuration constants for STT vosk model
VOSK_MODEL_PATH = "None"

# Configuration constants for the API client
OLLAMA_API_URL = "http://localhost:11434/api/generate"

if LANGUAGE == Language.ITALIAN:
    # Modelli da utilizzare (sostituisci GPT2 con un modello italiano o multilingue)
    MODEL_TALK = "GroNLP/gpt2-small-italian"  # o un altro modello adatto all'italiano
    MODEL_THINK = "GroNLP/gpt2-small-italian"
else:
    MODEL_TALK = "gpt2"
    MODEL_THINK = "gpt2"

# Configuration constants for the speech recognizer
WAKE_WORD = 'Hello'
LISTEN_TIMEOUT = 10
WAKE_SOUND = 'sounds/wake_up.mp3'
STOP_SOUND = 'sounds/stop.mp3'
TIMEOUT_SOUND = 'sounds/stop.mp3'

if ARCHITECTURE == Architecture.AARCH64:
   WAKE_SOUND = 'sounds/wake_up.wav'
   STOP_SOUND = 'sounds/stop.wav'
   TIMEOUT_SOUND = 'sounds/stop.wav'

# switch language
if LANGUAGE == Language.ENGLISH: 
    VOSK_MODEL_PATH = "recognizer/models/vosk-model-small-en-us-0.15"
    VOICE = "mb-us1" # Select MBROLA-voices, example: mb-us1 (american english female voice); mb-it4 (italian female voice)
elif LANGUAGE == Language.ITALIAN:
    VOSK_MODEL_PATH = "recognizer/models/vosk-model-small-it-0.22"
    VOICE = "mb-it4" # Select MBROLA-voices, example: mb-us1 (american english female voice); mb-it4 (italian female voice) 

