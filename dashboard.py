import streamlit as st
import cv2
import mediapipe as mp
import pickle
import pyttsx3
import speech_recognition as sr
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sign Language AI",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 Sign Language Communication Dashboard")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("gesture_model.pkl", "rb"))

# ---------------- TEXT TO SPEECH ----------------
engine = pyttsx3.init()
engine.setProperty('rate', 250)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
draw = mp.solutions.drawing_utils

# ---------------- SESSION STATE ----------------
if "sentence" not in st.session_state:
    st.session_state.sentence = []

if "current_gesture" not in st.session_state:
    st.session_state.current_gesture = ""

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

if "cap" not in st.session_state:
    st.session_state.cap = None

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([3,1])

# ---------------- CAMERA PANEL ----------------
with col1:
    st.subheader("📷 Live Camera Feed")
    FRAME_WINDOW = st.empty()

# ---------------- CONTROLS ----------------
with col2:
    st.subheader("🖐 Detected Gesture")

    if st.button("📷 Start Camera"):
        st.session_state.camera_on = True
        st.session_state.cap = cv2.VideoCapture(0)

    if st.button("🛑 Stop Camera"):
        st.session_state.camera_on = False

    # ✅ ADD GESTURE + AUTO SPEAK WORD
    if st.button("✅ Add Gesture"):
        if st.session_state.current_gesture not in ["None", ""]:
            st.session_state.sentence.append(st.session_state.current_gesture)

            # 🔊 SPEAK FULL WORD
            full_text = "".join(st.session_state.sentence)
            speak(full_text)

    # ✅ SPACE BUTTON
    if st.button("␣ Add Space"):
        st.session_state.sentence.append(" ")

    gesture_display = st.empty()

# ---------------- CAMERA LOOP ----------------
if st.session_state.camera_on:

    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)

    cap = st.session_state.cap

    while st.session_state.camera_on:

        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        prediction = "None"

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                data = []
                for lm in hand.landmark:
                    data.extend([lm.x, lm.y])

                prediction = model.predict([data])[0]

                draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                cv2.putText(frame, prediction, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

        st.session_state.current_gesture = prediction

        FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)
        gesture_display.metric("Detected Gesture", prediction)

        time.sleep(0.03)

        if not st.session_state.camera_on:
            break

    cap.release()
    st.session_state.cap = None

# ---------------- SENTENCE DISPLAY ----------------
st.subheader("📝 Sentence")
st.write("".join(st.session_state.sentence))

# ---------------- CONTROLS ----------------
colA, colB = st.columns(2)

with colA:
    if st.button("🔊 Speak Sentence"):
        speak("".join(st.session_state.sentence))

with colB:
    if st.button("🗑 Clear Sentence"):
        st.session_state.sentence.clear()

# ---------------- TEXT TO SPEECH ----------------
st.subheader("🔊 Text to Speech")

text_input = st.text_input("Enter text")

if st.button("Speak Text"):
    if text_input:
        speak(text_input)

# ---------------- SPEECH TO TEXT ----------------
st.subheader("🎤 Speech to Text")

if st.button("Start Listening"):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Listening... Speak now")
        recognizer.adjust_for_ambient_noise(source)

        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            st.success("You said: " + text)
        except:
            st.error("Speech recognition failed")
