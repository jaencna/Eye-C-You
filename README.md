# 👁️ Eye-C-You  
**A Real-Time Eyelid Detection System for Preventing Driver Drowsiness Using Arduino and Python**

---

## 🧠 Overview

**Eye-C-You** is a real-time driver drowsiness detection system that monitors eyelid activity to help prevent road accidents caused by fatigue. By using an infrared (IR) sensor or eye-blink detection module, the system detects signs of prolonged eye closure — a key indicator of drowsiness. Once detected, it triggers an audible buzzer to alert the driver.

While the project is currently not yet a fully standalone embedded system, it combines **Arduino hardware** and **Python software** to demonstrate a proof-of-concept solution for road safety. Future versions aim to operate independently without requiring a PC.

This system is especially relevant in the **Philippines**, where long driving hours, fatigue, and limited vehicle monitoring contribute significantly to road-related fatalities.

---

## 🚘 Key Features

- 🔍 **Real-time eyelid monitoring** via IR or eye-blink sensor  
- 🤖 **Arduino-controlled alert logic** with Python integration  
- 🔊 **Buzzer alert system** to wake drowsy drivers immediately  
- ⚙️ **Customizable blink threshold** for detection sensitivity  
- 💡 **LED indicators** for system status and blink detection feedback  

---

## 🛠️ Technology Stack

| Component            | Description |
|----------------------|-------------|
| **Arduino Uno / Nano** | Handles sensor input and triggers alerts |
| **IR Sensor / Eye Blink Module** | Detects eye closure or blinking |
| **Python + OpenCV (Planned / Experimental)** | Enables webcam-based detection and future facial landmark tracking |
| **Buzzer** | Provides immediate audio alert |
| **LEDs** | Visual indicators for blink detection |
| **PC or Laptop** | Currently used to run Python scripts and interface with Arduino |

---

## 📌 Use Cases

- 🚐 Public utility drivers (jeepney, tricycle, taxi)  
- 🚛 Long-haul and delivery truck drivers  
- 🏫 Driving schools (for safety awareness and training)  
- 🚗 Frequent travelers on personal vehicles  

---

## 📈 Future Development

- ✅ Integrate **Python + OpenCV** for advanced webcam-based eye tracking  
- ✅ Optimize sensitivity for different eye-blink patterns  
- 🔄 Transition to a **Raspberry Pi** for a fully embedded and portable system  
- 🧠 Add machine learning for smarter drowsiness prediction  
- 📊 Add logging/reporting features for fleet or driver monitoring  

---

## ⚠️ Disclaimer

This project is a **proof of concept** developed for demonstration purposes and is **not yet certified for use in actual vehicles**. Use responsibly in controlled environments only.

---



![image](https://github.com/user-attachments/assets/bfc1900b-b4e8-4b32-b0a7-7b209ea7a3a0)

![image](https://github.com/user-attachments/assets/d65665ea-c297-4420-a9d3-ae59b5dae5f6)

![image](https://github.com/user-attachments/assets/329cad5b-09bf-4a52-b423-c4316d68e0a8)

![image](https://github.com/user-attachments/assets/14ee41c4-2490-4590-813b-6edb6bc93f0d)
