import time
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

MODEL_DIR = r"D:\Apps\CorridaCachorro\models\vosk-model-small-es-0.42\vosk-model-small-es-0.42"

print("Devices (resumo):")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        print(i, d["name"])

print("\nDefault device:", sd.default.device)

m = Model(MODEL_DIR)
r = KaldiRecognizer(m, 16000)

def cb(indata, frames, time_info, status):
    if status:
        print("STATUS:", status)
        return

    data = bytes(indata)  # ✅ CORRETO no RawInputStream

    if r.AcceptWaveform(data):
        text = json.loads(r.Result()).get("text", "")
        if text:
            print(">>", text)

print("\nFale agora por 10 segundos...")
with sd.RawInputStream(
    samplerate=16000,
    blocksize=8000,
    dtype="int16",
    channels=1,
    callback=cb
):
    time.sleep(10)

print("Fim.")
