import tkinter as tk
from tkinter import ttk
import random
import time
import threading
from collections import deque
import os
import json
import queue

# Camera / Gestos
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Voz (Vosk)
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# -----------------------------
# Config do jogo
# -----------------------------
W, H = 900, 420
TRACK_Y1 = 120
TRACK_Y2 = 260
FINISH_X = W - 120
FPS_MS = 16

HAND_MODEL_PATH = os.path.join("models", "hand_landmarker.task")
VOSK_BASE_DIR = os.path.join("models")  # vamos procurar um modelo dentro de models/


class DogRaceGame:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🐶 Corrida de Cachorro (Mão + Voz)")
        self.root.resizable(False, False)

        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # UI
        self.canvas = tk.Canvas(root, width=W, height=H, bg="#0b0f12", highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=3, padx=14, pady=(14, 10))

        self.btn_start = ttk.Button(root, text="▶ Iniciar", command=self.start)
        self.btn_start.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))

        self.btn_reset = ttk.Button(root, text="↩ Reset", command=self.reset)
        self.btn_reset.grid(row=1, column=1, sticky="ew", padx=7, pady=(0, 14))

        self.btn_quit = ttk.Button(root, text="✖ Sair", command=self.on_close)
        self.btn_quit.grid(row=1, column=2, sticky="ew", padx=14, pady=(0, 14))

        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        root.grid_columnconfigure(2, weight=1)

        # Teclado (backup)
        root.bind("<KeyPress-Return>", lambda e: self.start())
        root.bind("<space>", lambda e: self.boost())
        root.bind("<KeyPress-s>", lambda e: self.stop())
        root.bind("<KeyPress-r>", lambda e: self.reset())

        # Estado jogo
        self.running = False
        self.winner = None
        self.last_time = time.time()

        self.dog_speed = 170.0
        self.enemy_speed = 165.0

        self._build_scene()
        self.reset()

        # -----------------------------
        # Gestos (mão)
        # -----------------------------
        self.gesture_queue = deque(maxlen=12)
        self.last_gesture = None
        self.last_gesture_time = 0.0
        self.gesture_cooldown = 0.55
        self.hand_thread_running = True
        self._start_hand_thread()

        # -----------------------------
        # Voz (Vosk)
        # -----------------------------
        self.voice_queue = queue.Queue()
        self.voice_running = True

        self.voice_last_cmd = None
        self.voice_last_time = 0.0
        self.voice_cooldown = 0.9

        self.vosk_model_dir = self._find_vosk_model_dir()
        print("VOSK model dir =", self.vosk_model_dir)

        if self.vosk_model_dir:
            self._start_voice_thread()
        else:
            print("VOSK: Modelo não encontrado dentro de models/ (ex: models/vosk-model-small-xx)")

        # Loops
        self.root.after(FPS_MS, self.loop)
        self.root.after(120, self._poll_hand_gestures)
        self.root.after(150, self._poll_voice_commands)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -----------------------------
    # UI / Cena
    # -----------------------------
    def _build_scene(self):
        self.canvas.create_text(
            18, 18, anchor="w", fill="white",
            font=("Segoe UI", 14, "bold"),
            text="🐶 Corrida | Mão: ✋ start ✊ stop 3 dedos turbo ✌ reset | Voz: empezar/alto/turbo/reset (ES melhor)"
        )

        self.canvas.create_rectangle(40, TRACK_Y1 - 40, W - 40, TRACK_Y1 + 40,
                                     fill="#121821", outline="#1f2a36", width=2)
        self.canvas.create_rectangle(40, TRACK_Y2 - 40, W - 40, TRACK_Y2 + 40,
                                     fill="#121821", outline="#1f2a36", width=2)

        self.canvas.create_rectangle(FINISH_X, 70, FINISH_X + 10, H - 70,
                                     fill="#f6d55c", outline="")
        self.canvas.create_text(FINISH_X + 5, 60, text="CHEGADA",
                                fill="#f6d55c", font=("Segoe UI", 10, "bold"))

        self.hud = self.canvas.create_text(
            W - 20, 18, anchor="e",
            fill="#cfe8ff", font=("Segoe UI", 12, "bold"),
            text="Pronto."
        )

        self.dog = self.canvas.create_text(70, TRACK_Y1, text="🐶", font=("Segoe UI Emoji", 28))
        self.enemy = self.canvas.create_text(70, TRACK_Y2, text="🐕", font=("Segoe UI Emoji", 28))

        self.dog_shadow = self.canvas.create_oval(52, TRACK_Y1 + 22, 88, TRACK_Y1 + 32,
                                                  fill="#000000", outline="", stipple="gray50")
        self.enemy_shadow = self.canvas.create_oval(52, TRACK_Y2 + 22, 88, TRACK_Y2 + 32,
                                                    fill="#000000", outline="", stipple="gray50")

        self.center_msg = self.canvas.create_text(
            W / 2, H / 2, text="",
            fill="white", font=("Segoe UI", 24, "bold")
        )

    # -----------------------------
    # Jogo
    # -----------------------------
    def reset(self):
        self.running = False
        self.winner = None
        self.last_time = time.time()

        self.dog_x = 70.0
        self.enemy_x = 70.0

        self.turbo = 0.0
        self.turbo_decay = 220.0
        self.max_turbo = 260.0

        self.enemy_jitter = 0.0

        self._place(self.dog, self.dog_shadow, self.dog_x, TRACK_Y1)
        self._place(self.enemy, self.enemy_shadow, self.enemy_x, TRACK_Y2)

        self.canvas.itemconfig(self.center_msg, text="")
        self.canvas.itemconfig(self.hud, text="Pronto. ✋ ou 'empezar' para iniciar.")

    def start(self):
        if self.winner:
            return
        self.running = True
        self.last_time = time.time()
        self.canvas.itemconfig(self.hud, text="Correndo...")

    def stop(self):
        self.running = False
        if not self.winner:
            self.canvas.itemconfig(self.hud, text="Pausado.")

    def boost(self):
        if self.winner:
            return
        self.turbo = min(self.max_turbo, self.turbo + 170.0)
        self.canvas.itemconfig(self.hud, text="TURBO! 🚀")

    def loop(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        if self.running and not self.winner:
            if self.turbo > 0:
                self.turbo = max(0.0, self.turbo - self.turbo_decay * dt)

            self.enemy_jitter += random.uniform(-25, 25) * dt
            self.enemy_jitter = max(-35, min(35, self.enemy_jitter))

            self.dog_x += (self.dog_speed + self.turbo) * dt
            self.enemy_x += (self.enemy_speed + self.enemy_jitter) * dt

            self.dog_x = min(self.dog_x, FINISH_X - 10)
            self.enemy_x = min(self.enemy_x, FINISH_X - 10)

            self._place(self.dog, self.dog_shadow, self.dog_x, TRACK_Y1)
            self._place(self.enemy, self.enemy_shadow, self.enemy_x, TRACK_Y2)

            if self.dog_x >= FINISH_X - 10 or self.enemy_x >= FINISH_X - 10:
                self.running = False
                if self.dog_x >= FINISH_X - 10 and self.enemy_x >= FINISH_X - 10:
                    self.winner = "empate"
                    self.canvas.itemconfig(self.center_msg, text="EMPATE! 🟰")
                elif self.dog_x >= FINISH_X - 10:
                    self.winner = "voce"
                    self.canvas.itemconfig(self.center_msg, text="VOCÊ VENCEU! 🏆🐶")
                else:
                    self.winner = "inimigo"
                    self.canvas.itemconfig(self.center_msg, text="VOCÊ PERDEU! 😅🐕")

                self.canvas.itemconfig(self.hud, text="Fim. ✌ ou 'reset' para reiniciar.")
            else:
                pct_you = int((self.dog_x / (FINISH_X - 10)) * 100)
                pct_ai = int((self.enemy_x / (FINISH_X - 10)) * 100)
                self.canvas.itemconfig(self.hud, text=f"Você: {pct_you}% | Rival: {pct_ai}% | Turbo: {int(self.turbo)}")

        self.root.after(FPS_MS, self.loop)

    def _place(self, dog_item, shadow_item, x, y):
        self.canvas.coords(dog_item, x, y)
        self.canvas.coords(shadow_item, x - 18, y + 22, x + 18, y + 32)

    # -----------------------------
    # Gestos por mão (MediaPipe Tasks)
    # -----------------------------
    def _start_hand_thread(self):
        t = threading.Thread(target=self._hand_loop, daemon=True)
        t.start()

    def _open_camera(self):
        for idx in (0, 1, 2):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                return cap, idx
        return None, None

    def _hand_loop(self):
        if not os.path.exists(HAND_MODEL_PATH):
            print("ERRO: modelo de mão não encontrado:", HAND_MODEL_PATH)
            return

        base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1
        )
        landmarker = mp_vision.HandLandmarker.create_from_options(options)

        cap, cam_idx = self._open_camera()
        if cap is None:
            print("ERRO: Nenhuma câmera abriu. Verifique permissões do Windows.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        start_time = time.time()
        print(f"Camera OK (index={cam_idx}).")

        while self.hand_thread_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int((time.time() - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            gesture = None
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                lm = result.hand_landmarks[0]
                gesture = self._classify_gesture(lm)

            if gesture:
                self.gesture_queue.append(gesture)

            cv2.putText(frame, f"Gesture: {gesture or '-'}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 255, 60), 2)
            cv2.imshow("Hand Control (Q para sair)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _classify_gesture(self, lm):
        def ext(tip_id, pip_id):
            return lm[tip_id].y < lm[pip_id].y

        thumb = ext(4, 3)
        idx = ext(8, 6)
        mid = ext(12, 10)
        ring = ext(16, 14)
        pink = ext(20, 18)

        count = sum([thumb, idx, mid, ring, pink])

        palm_y = lm[9].y
        tips_y = [lm[8].y, lm[12].y, lm[16].y, lm[20].y]
        avg_tips_y = sum(tips_y) / len(tips_y)

        if count <= 1 or (avg_tips_y > palm_y and count <= 2):
            return "fist"

        if count >= 4:
            return "open_palm"

        if idx and mid and (not ring) and (not pink):
            return "peace"

        # ✅ 3 dedos = turbo (indicador + medio + anelar)
        if idx and mid and ring and (not thumb) and (not pink):
            return "three_fingers"

        return None

    def _poll_hand_gestures(self):
        if not self.gesture_queue:
            self.root.after(120, self._poll_hand_gestures)
            return

        N = 6
        recent = list(self.gesture_queue)[-N:]
        if not recent:
            self.root.after(120, self._poll_hand_gestures)
            return

        gesture = max(set(recent), key=recent.count)
        cnt = recent.count(gesture)

        if cnt < 4:
            self.root.after(120, self._poll_hand_gestures)
            return

        now = time.time()
        if (gesture != self.last_gesture) or (now - self.last_gesture_time > self.gesture_cooldown):
            if gesture == "open_palm":
                self.start()
                self.canvas.itemconfig(self.hud, text="✋ START (mão)")
            elif gesture == "fist":
                self.stop()
                self.canvas.itemconfig(self.hud, text="✊ STOP (mão)")
            elif gesture == "three_fingers":
                self.boost()
                self.canvas.itemconfig(self.hud, text="3️⃣ TURBO (mão)")
            elif gesture == "peace":
                self.reset()
                self.canvas.itemconfig(self.hud, text="✌ RESET (mão)")

            self.last_gesture = gesture
            self.last_gesture_time = now

        self.root.after(120, self._poll_hand_gestures)

    # -----------------------------
    # Voz (Vosk) - robusto com auto-selecao de microfone
    # -----------------------------
    def _find_vosk_model_dir(self):
        for name in os.listdir(VOSK_BASE_DIR):
            p = os.path.join(VOSK_BASE_DIR, name)
            if os.path.isdir(p) and all(os.path.exists(os.path.join(p, x)) for x in ("am", "conf", "graph")):
                return p

        for name in os.listdir(VOSK_BASE_DIR):
            p = os.path.join(VOSK_BASE_DIR, name)
            if not os.path.isdir(p):
                continue
            for sub in os.listdir(p):
                p2 = os.path.join(p, sub)
                if os.path.isdir(p2) and all(os.path.exists(os.path.join(p2, x)) for x in ("am", "conf", "graph")):
                    return p2

        vosk_dir = os.path.join("models", "vosk")
        if os.path.isdir(vosk_dir):
            for name in os.listdir(vosk_dir):
                p = os.path.join(vosk_dir, name)
                if os.path.isdir(p) and all(os.path.exists(os.path.join(p, x)) for x in ("am", "conf", "graph")):
                    return p

        return None

    def _pick_input_device(self):
        devices = sd.query_devices()
        candidates = []
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                name = (d.get("name") or "").lower()
                score = 0
                if "micro" in name or "mic" in name:
                    score += 10
                if "realtek" in name:
                    score += 2
                candidates.append((score, i, d))
        if not candidates:
            return None, None
        candidates.sort(reverse=True, key=lambda x: x[0])
        score, idx, dev = candidates[0]
        return idx, dev

    def _start_voice_thread(self):
        t = threading.Thread(target=self._voice_loop, daemon=True)
        t.start()
        print("Voz ligada! (modelo ES) Fale: empezar / alto / turbo / reset")

    def _text_to_command(self, text: str):
        t = text.lower().strip()

        if any(w in t for w in ["empezar", "arranca", "vamos", "start", "corre", "iniciar", "vai"]):
            return "START"
        if any(w in t for w in ["alto", "parar", "stop", "para", "pare", "detener"]):
            return "STOP"
        if any(w in t for w in ["turbo", "rapido", "rápido", "acelera", "acelerar", "mas rapido", "más rápido"]):
            return "TURBO"
        if any(w in t for w in ["reset", "reinicio", "reiniciar", "recomeçar", "recomecar"]):
            return "RESET"

        return None

    def _voice_loop(self):
        try:
            if not self.vosk_model_dir:
                print("ERRO VOZ: vosk_model_dir vazio")
                return

            dev_idx, dev_info = self._pick_input_device()
            if dev_idx is None:
                print("ERRO VOZ: nenhum dispositivo de entrada encontrado")
                return

            sr = int(dev_info.get("default_samplerate", 16000))
            print(f"VOICE device={dev_idx} name={dev_info.get('name')}")
            print("VOICE samplerate =", sr)

            model = Model(self.vosk_model_dir)
            rec = KaldiRecognizer(model, sr)

            def callback(indata, frames, time_info, status):
                if status:
                    print("VOICE STATUS:", status)
                    return

                data = bytes(indata)

                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    text = (res.get("text") or "").strip().lower()
                    if text:
                        print("VOICE FINAL:", text)
                        cmd = self._text_to_command(text)
                        if cmd:
                            print("VOICE CMD:", cmd)
                            self.voice_queue.put(cmd)
                else:
                    pres = json.loads(rec.PartialResult())
                    p = (pres.get("partial") or "").strip().lower()
                    if p and len(p.split()) <= 2:
                        cmd = self._text_to_command(p)
                        if cmd:
                            self.voice_queue.put(cmd)

            with sd.RawInputStream(
                device=dev_idx,
                samplerate=sr,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=callback,
            ):
                while self.voice_running:
                    time.sleep(0.1)

        except Exception as e:
            print("ERRO VOZ:", e)

    def _poll_voice_commands(self):
        try:
            while True:
                cmd = self.voice_queue.get_nowait()
                now = time.time()

                if self.voice_last_cmd == cmd and (now - self.voice_last_time) < self.voice_cooldown:
                    continue

                if cmd == "START":
                    self.start()
                    self.canvas.itemconfig(self.hud, text="🎤 START")
                elif cmd == "STOP":
                    self.stop()
                    self.canvas.itemconfig(self.hud, text="🎤 STOP")
                elif cmd == "TURBO":
                    self.boost()
                    self.canvas.itemconfig(self.hud, text="🎤 TURBO")
                elif cmd == "RESET":
                    self.reset()
                    self.canvas.itemconfig(self.hud, text="🎤 RESET")

                self.voice_last_cmd = cmd
                self.voice_last_time = now

        except queue.Empty:
            pass

        self.root.after(150, self._poll_voice_commands)

    def on_close(self):
        self.hand_thread_running = False
        self.voice_running = False
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    DogRaceGame(root)
    root.mainloop()
