import os
import logging
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import csv
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger("tensorflow").setLevel(logging.FATAL)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

USER_MAP_FILE = "user_map.json"


# ----------------- User Management -----------------
def load_user_map():
    if os.path.exists(USER_MAP_FILE):
        with open(USER_MAP_FILE, "r") as f:
            return json.load(f)
    return {}


def save_user_map(user_map):
    with open(USER_MAP_FILE, "w") as f:
        json.dump(user_map, f, indent=2)


def get_next_user_id():
    user_map = load_user_map()
    if not user_map:
        return 1
    existing_ids = sorted(int(uid) for uid in user_map.keys())
    return max(existing_ids) + 1


def cleanup_user_map(data_dir="data"):
    user_map = load_user_map()
    if not os.path.exists(data_dir):
        save_user_map({})
        return {}

    valid_ids = set()
    for file in os.listdir(data_dir):
        if file.startswith("user.") and file.endswith(".jpg"):
            try:
                uid = file.split(".")[1]
                valid_ids.add(uid)
            except:
                continue

    cleaned_map = {uid: name for uid, name in user_map.items() if uid in valid_ids}
    save_user_map(cleaned_map)
    return cleaned_map


def delete_user(user_id, data_dir="data"):
    user_map = load_user_map()

    if str(user_id) not in user_map:
        return False, f"⚠️ User {user_id} not found in mapping."

    deleted_files = 0
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.startswith(f"user.{user_id}."):
                os.remove(os.path.join(data_dir, file))
                deleted_files += 1

    csv_path = f"landmarks/user_{user_id}.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)

    user_name = user_map.pop(str(user_id))
    save_user_map(user_map)

    trained = train_classifier(data_dir)
    if trained:
        return True, f"🗑️ Deleted {deleted_files} images + entry for {user_name} (ID {user_id}). Model retrained."
    else:
        return True, f"🗑️ Deleted {deleted_files} images + entry for {user_name} (ID {user_id}). (No data left to retrain)"


# ----------------- Detection & Recognition -----------------
def detect(frame, faceCascade, img_id, user_id, face_mesh, csv_writer):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    saved = False
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        os.makedirs("data", exist_ok=True)
        cv2.imwrite(f"data/user.{user_id}.{img_id}.jpg", face_img)
        saved = True

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
            )
            h, w, _ = frame.shape
            row = [user_id, img_id]
            for lm in face_landmarks.landmark:
                row.append(int(lm.x * w))
                row.append(int(lm.y * h))
            csv_writer.writerow(row)

    return frame, saved


def recognize(frame, clf, faceCascade, face_mesh, user_map, threshold=70):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    for (x, y, w, h) in faces:
        try:
            face_img = gray[y:y+h, x:x+w]
            id_, conf = clf.predict(face_img)

            if conf < threshold and str(id_) in user_map:
                name = user_map[str(id_)]
                label = f"{name}"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            center = (x + w//2, y + h//2)
            radius = int(max(w, h)/2)
            cv2.circle(frame, center, radius, color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except:
            cv2.putText(frame, "Unknown", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
            )

    return frame


# ----------------- Training -----------------
def train_classifier(data_dir="data"):
    faces, ids = [], []
    if not os.path.exists(data_dir):
        return False

    for file in os.listdir(data_dir):
        if file.endswith(".jpg"):
            path = os.path.join(data_dir, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            try:
                id_ = int(file.split(".")[1])
            except:
                continue
            faces.append(img)
            ids.append(id_)

    if len(faces) == 0:
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.save("classifier.yml")
    return True


# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Face Recognition", page_icon="🧑")
st.title("🧑 Real Time Face Recognition")

menu = [
    "📸 Capture Dataset",
    "🧠 Train Model",
    "🔍 Recognize Face",
    "🗑️ Delete User",
    "📋 View Registered Users",
    "🖼️ Upload Image for Recognition",
    "📊 View User Statistics",
    "🔄 Update User Name"
]
choice = st.sidebar.selectbox("Menu", menu)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
user_map = load_user_map()
clf = None
if os.path.exists("classifier.yml"):
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.yml")


# ----------------- Features -----------------
if choice == "📸 Capture Dataset":
    suggested_id = get_next_user_id()
    st.info(f"Next available User ID is {suggested_id}")
    user_id = st.number_input("Enter User ID", min_value=suggested_id, value=suggested_id, step=1)
    user_name = st.text_input("Enter User Name")

    if st.button("Start Capture"):
        if not user_name.strip():
            st.error("⚠️ Please enter a valid name!")
        else:
            user_map[str(user_id)] = user_name
            save_user_map(user_map)

            cap = cv2.VideoCapture(0)
            saved_count = 0
            stframe = st.empty()
            os.makedirs("landmarks", exist_ok=True)
            csv_file = open(f"landmarks/user_{user_id}.csv", mode="w", newline="")
            csv_writer = csv.writer(csv_file)

            with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
                while saved_count < 50:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from webcam.")
                        break
                    frame, saved = detect(frame, faceCascade, saved_count, user_id, face_mesh, csv_writer)
                    if saved:
                        saved_count += 1
                    stframe.image(frame, channels="BGR", caption=f"Saved Images: {saved_count}/50")

            csv_file.close()
            cap.release()
            cv2.destroyAllWindows()
            st.success(f"✅ Dataset captured for {user_name} (ID: {user_id})")


elif choice == "🧠 Train Model":
    if st.button("Train Now"):
        trained = train_classifier("data")
        if trained:
            clf = cv2.face.LBPHFaceRecognizer_create()
            clf.read("classifier.yml")
            st.success("✅ Model trained and saved successfully!")
        else:
            st.error("❌ No valid data available to train.")

elif choice == "🔍 Recognize Face":
    if clf is None or len(user_map) == 0:
        st.warning("⚠️ No trained model or users available. Please capture dataset and train first.")
    elif st.button("Start Recognition"):
        st.info("🔎 Recognition started. Close Streamlit to stop.")
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Webcam error.")
                    break
                frame = recognize(frame, clf, faceCascade, face_mesh, user_map)
                stframe.image(frame, channels="BGR")
        cap.release()
        cv2.destroyAllWindows()

elif choice == "🗑️ Delete User":
    del_id = st.number_input("Enter User ID to Delete", min_value=1, step=1)
    if st.button("Delete User"):
        success, msg = delete_user(del_id)
        if success:
            st.success(msg)
        else:
            st.error(msg)

elif choice == "📋 View Registered Users":
    if len(user_map) == 0:
        st.info("No users registered yet.")
    else:
        data = []
        for uid, name in user_map.items():
            count = len([f for f in os.listdir("data") if f.startswith(f"user.{uid}.")])
            data.append({"User ID": uid, "Name": name, "Images": count})
        st.table(data)

elif choice == "🖼️ Upload Image for Recognition":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            frame = recognize(frame, clf, faceCascade, face_mesh, user_map)
        st.image(frame, channels="BGR")

elif choice == "📊 View User Statistics":
    total_users = len(user_map)
    total_images = len(os.listdir("data")) if os.path.exists("data") else 0
    st.metric("Total Registered Users", total_users)
    st.metric("Total Images in Dataset", total_images)
    if total_users > 0:
        for uid, name in user_map.items():
            count = len([f for f in os.listdir("data") if f.startswith(f"user.{uid}.")])
            st.write(f"**{name} (ID: {uid})** → {count} images")

elif choice == "🔄 Update User Name":
    upd_id = st.number_input("Enter User ID to Update", min_value=1, step=1)
    new_name = st.text_input("Enter New Name")
    if st.button("Update Name"):
        if str(upd_id) in user_map:
            old_name = user_map[str(upd_id)]
            user_map[str(upd_id)] = new_name
            save_user_map(user_map)
            st.success(f"✅ Updated user {old_name} (ID {upd_id}) → {new_name}")
        else:
            st.error("⚠️ User ID not found.")
