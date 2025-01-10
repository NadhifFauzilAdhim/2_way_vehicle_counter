# 🚗 2-Way Vehicle Counter 🚦  

**2-Way Vehicle Counter** is a Python-based application designed to count vehicles passing through two lanes (up and down directions) in real-time. It utilizes YOLO for object detection, providing a practical tool for traffic monitoring and data collection.  

---

## ✨ Features  
- **🚙 Real-Time Vehicle Detection:**  
  Uses YOLO for accurate and efficient vehicle detection and tracking.  

- **🔄 Dual-Lane Counting:**  
  Counts vehicles passing through two separate lanes (up and down directions).  

- **📊 Data Visualization:**  
  Displays live updates in text, table, and bar chart formats.  

- **💾 Data Storage:**  
  Saves vehicle count data (categorized by direction and type) in an SQLite database.  

- **⚙️ User-Friendly Interface:**  
  Built with Streamlit for easy configuration and visualization.  

---

## 🛠️ Technologies Used  
- **OpenCV:** Captures video feed and processes frames.  
- **YOLO (Ultralytics):** Performs object detection.  
- **Streamlit:** Provides an interactive web-based user interface.  
- **SQLite:** Stores vehicle count data.  
- **Torch:** Powers the YOLO model.  

---

## 🚀 How It Works  
1. **Initialization:**  
   - Loads the YOLO model (`yolov8l.pt`) and sets up the SQLite database.  
   - Configures the webcam resolution based on user input.  

2. **Vehicle Detection:**  
   - Processes video frames from the webcam using YOLO.  
   - Detects vehicles and annotates frames with bounding boxes and labels.  

3. **Counting Logic:**  
   - Monitors vehicles crossing trigger lines in the video.  
   - Updates counts based on vehicle category and direction (up/down).  

4. **Visualization:**  
   - Displays the video feed with annotations in real-time.  
   - Updates the count data in text, table, and bar chart formats.  

5. **Reset Functionality:**  
   - Users can reset the vehicle counts via a button in the sidebar.  

---

## 📦 Installation  
1. Clone this repository:  
   ```bash
   git clone https://github.com/NadhifFauzilAdhim/2_way_vehicle_counter.git
   ```
   ```bash
   cd 2-way-vehicle-counter
    ```
    ```bash
    pip install -r requirements.txt
    ```
    ```bash
    streamlit run app.py
    ```

## 🖼️ Application Preview

    2-way-vehicle-counter/
    │
    ├── db/
    │   └── vehicle_counter.db  # SQLite database
    ├── model/
    │   └── yolov8l.pt          # YOLO model file
    ├── app.py                  # Main application script
    ├── requirements.txt        # Dependency list
    └── README.md               # Project documentation
    

    


