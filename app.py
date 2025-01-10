import cv2
import streamlit as st
from ultralytics import YOLO
import supervision as sv
import numpy as np
import torch
import pandas as pd
import sqlite3

TRIGGER_LINE_TOP_Y = 0.5
TRIGGER_LINE_BOTTOM_Y = 0.5

# Database setup
DB_PATH = "db/vehicle_counter.db"

def create_tables():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_counts (
            direction TEXT,
            category TEXT,
            count INTEGER,
            PRIMARY KEY (direction, category)
        )
        """)
        conn.commit()

def reset_counts():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM vehicle_counts")
        conn.commit()

def update_count(direction, category, increment=1):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO vehicle_counts (direction, category, count)
        VALUES (?, ?, ?)
        ON CONFLICT(direction, category) DO UPDATE SET count = count + ?
        """, (direction, category, increment, increment))
        conn.commit()

def get_counts():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vehicle_counts")
        return cursor.fetchall()

def main():
    st.title("2-way Road Counter")
    create_tables()
    st.sidebar.header("Settings")
    webcam_resolution = st.sidebar.selectbox(
        "Webcam Resolution",
        options=[(1280, 720), (640, 480)],
        index=0
    )

    if st.sidebar.button("Reset Counts"):
        reset_counts()

    frame_width, frame_height = webcam_resolution

    cap = cv2.VideoCapture(0) # video
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.write(f"Using device: {device}")

    model = YOLO("model/yolov8l.pt").to(device)

    box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=0.2
    )

    stframe = st.empty()
    count_placeholder = st.empty()
    chart_placeholder = st.empty()
    table_placeholder = st.empty()

    line_top_y = int(TRIGGER_LINE_TOP_Y * frame_height)
    line_bottom_y = int(TRIGGER_LINE_BOTTOM_Y * frame_height)
    line_half_width = frame_width // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame from webcam")
            break

        def letterbox_image(img, desired_size=(640, 640)):
            ih, iw = img.shape[:2]
            h, w = desired_size
            scale = min(w / iw, h / ih)
            nw, nh = int(iw * scale), int(ih * scale)

            resized_img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

            new_img = np.full((h, w, 3), 128, dtype=np.uint8)
            top = (h - nh) // 2
            left = (w - nw) // 2
            new_img[top:top + nh, left:left + nw, :] = resized_img
            return new_img, scale, left, top

        input_frame, scale, pad_left, pad_top = letterbox_image(frame, (640, 640))

        frame_tensor = (
            torch.from_numpy(input_frame)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
            .float()
        ) / 255.0

        # Perform detection
        results = model(frame_tensor)
        detections = sv.Detections.from_yolov8(results[0])

        # Rescale bounding boxes to original frame dimensions
        if detections.xyxy is not None:
            for i in range(len(detections.xyxy)):
                detections.xyxy[i][0::2] = (detections.xyxy[i][0::2] - pad_left) / scale
                detections.xyxy[i][1::2] = (detections.xyxy[i][1::2] - pad_top) / scale

        # Get class indices, confidences, and map to labels
        class_indices = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        labels = [f"{model.names[idx]} ({conf:.2f})" for idx, conf in zip(class_indices, confidences)]

        # Annotate the resized frame with detections and labels
        annotated_frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        # Draw trigger lines (half-width)
        cv2.line(annotated_frame, (0, line_top_y), (line_half_width, line_top_y), (0, 255, 0), 2)
        cv2.line(annotated_frame, (line_half_width, line_bottom_y), (frame_width, line_bottom_y), (0, 0, 255), 2)

        # Check detections crossing the lines
        if detections.xyxy is not None and len(detections.xyxy) > 0:
            for bbox, label in zip(detections.xyxy, labels):
                x1, y1, x2, y2 = bbox[:4]
                object_bottom = y2
                category_name = label.split(' ')[0]

                # Check for crossing top line (down to up, left side)
                if line_top_y - 5 < object_bottom < line_top_y + 5 and x2 <= line_half_width:
                    update_count("Up", category_name)

                # Check for crossing bottom line (up to down, right side)
                elif line_bottom_y - 5 < object_bottom < line_bottom_y + 5 and x1 >= line_half_width:
                    update_count("Down", category_name)

        # Update Streamlit components
        stframe.image(annotated_frame, channels="BGR")

        counts = get_counts()
        combined_counts = {f"{direction} - {category}": count for direction, category, count in counts}

        count_text = "### Vehicles Counted:\n"
        for direction, category, count in counts:
            count_text += f"- {direction}: {category} = {count}\n"
        
        count_placeholder.markdown(count_text)

        if combined_counts:
            chart_placeholder.bar_chart(combined_counts)
            df = pd.DataFrame(list(combined_counts.items()), columns=["Category", "Count"])
            table_placeholder.write(df)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
