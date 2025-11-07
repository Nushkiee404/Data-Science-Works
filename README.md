# ✋ Hand Gesture Volume Control 🎛️

A real-time **Computer Vision** project that controls your system volume using **hand gestures** detected through your webcam.  
Built with **Python, OpenCV, and Mediapipe**, this project blends AI and human-computer interaction in a fun and intuitive way.

---

## 🚀 Demo
![Demo](demo.gif)

---

## 🧠 Features
- ✋ Detects hand landmarks using **Mediapipe**
- 🔊 Controls system volume using **thumb–index finger distance**
- ⚡ Real-time performance (~30 FPS)
- 🧩 Customizable gestures for brightness or other actions
- 💻 Easy to set up and run locally

---

## 🛠️ Tech Stack
| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python |
| **Computer Vision** | OpenCV |
| **Hand Tracking** | Mediapipe |
| **Audio Control** | Pycaw |
| **Environment** | Jupyter / VS Code |

---

## 🧩 How It Works
1. The webcam feed is processed frame by frame using **OpenCV**
2. **Mediapipe** detects the position of hand landmarks
3. The distance between **thumb and index finger** is mapped to a **volume percentage**
4. The system volume dynamically adjusts in real time

---

## ⚙️ Setup and Installation

### 🔹 Step 1: Clone the Repository
```bash
git clone https://github.com/Nushkiee404/gesture-volume-control.git
cd gesture-volume-control

## 📂 Project Structure
gesture-volume-control/
│
├── hand_gesture_volume.py     # Main script
├── requirements.txt           # Dependencies
├── demo.gif                   # Demo animation
└── README.md                  # Documentation

---

## 🌟 Future Improvements 
- 🎮 Integrate gesture recognition with media playback  
- ☁️ Deploy as a real-time web app using Streamlit  

---

## 👩‍💻 Author
**Anushka Sharma**  
📧 Email: your-email@example.com  
🔗 [LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/Nushkiee404)

---

⭐ *If you like this project, don’t forget to star the repo!*

