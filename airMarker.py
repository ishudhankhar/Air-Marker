import cv2
import numpy as np
import os
import speech_recognition as sr
import pyttsx3  # Optional for voice feedback
import HandTrackingModule as htm

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech speed

# Function to recognize voice commands
def recognize_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for commands...")
        recognizer.adjust_for_ambient_noise(source)  # Reduce noise
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            print(f"Recognized: {command}")
            return command
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError:
            print("Speech service error")
        except sr.WaitTimeoutError:
            print("No voice input detected")
    return ""

# Function to provide voice feedback
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Folder containing toolbar images
folder_path = "Header"
myList = os.listdir(folder_path)
overlayList = [cv2.imread(f'{folder_path}/{imPath}') for imPath in myList]

# Set default values
header = cv2.resize(overlayList[0], (1280, 125))
drawColor = (255, 0, 255)
xp, yp = 0, 0
thickness = 15  # Default brush thickness
history = []  # Store strokes for undo/redo
redo_stack = []  # Store undone strokes

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detection_con=0.85)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Load previous drawing if available
if os.path.exists("saved_drawing.png"):
    imgCanvas = cv2.imread("saved_drawing.png")

# Command Mapping
def clear_canvas():
    global imgCanvas
    imgCanvas.fill(0)

def undo():
    global imgCanvas
    if history:
        redo_stack.append(history.pop())
        imgCanvas = np.zeros((720, 1280, 3), np.uint8)
        for stroke in history:
            cv2.line(imgCanvas, stroke[0], stroke[1], stroke[2], stroke[3])

def save_drawing():
    cv2.imwrite("saved_drawing.png", imgCanvas)

def exit_program():
    cap.release()
    cv2.destroyAllWindows()
    exit()

commands = {
    "clear canvas": clear_canvas,
    "undo": undo,
    "save drawing": save_drawing,
    "exit": exit_program,
}

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    
    # Add header toolbar image
    img[0:125, 0:1280] = header
    
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        
        fingers = detector.fingersup()
        
        # Selection mode (for color and eraser)
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 125:
                if 150 < x1 < 350:
                    header = cv2.resize(overlayList[0], (1280, 125))
                    drawColor = (255, 0, 255)
                elif 470 < x1 < 550:
                    header = cv2.resize(overlayList[1], (1280, 125))
                    drawColor = (0, 255, 0)
                elif 700 < x1 < 750:
                    header = cv2.resize(overlayList[2], (1280, 125))
                    drawColor = (0, 255, 255)
                elif 900 < x1 < 1000:
                    header = cv2.resize(overlayList[3], (1280, 125))
                    drawColor = (0, 0, 255)
                elif 1150 < x1 < 1200:
                    header = cv2.resize(overlayList[4], (1280, 125))
                    drawColor = (0, 0, 0)  # Eraser
        
        # Drawing mode
        elif fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 10, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
            history.append(((xp, yp), (x1, y1), drawColor, thickness))  # Store stroke
            redo_stack.clear()  # Reset redo stack
            xp, yp = x1, y1
        
        # Undo action
        if fingers == [1, 1, 1, 1, 1]:
            undo()
    
    key = cv2.waitKey(1)
    
    if key == ord('v'):  # Press 'V' to activate voice commands
        command = recognize_voice()
        if command in commands:
            commands[command]()  # Execute the corresponding function
            speak(f"Command executed: {command}")
    
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    
    cv2.imshow("Air Marker", img)
    
    if key == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
