import os
import logging
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import csv
import json
import matplotlib.cm as cm
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import hashlib

# ---------------- Admin Authentication ---------------- #
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_HASH = hashlib.sha256("admin123".encode()).hexdigest()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def authenticate(username, password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    if username == ADMIN_USERNAME and hashed_password == ADMIN_PASSWORD_HASH:
        st.session_state.authenticated = True
        st.success("Logged in as Admin!")
    else:
        st.error("Incorrect username or password.")
        st.session_state.authenticated = False

def logout():
    st.session_state.authenticated = False
    st.info("Logged out successfully.")
    st.rerun()

# ---------------- Setup ---------------- #
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger("tensorflow").setLevel(logging.FATAL)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

USER_MAP_FILE = "user_map.json"
COUNTRY_CODES = {
    "India": "+91", "United States": "+1", "United Kingdom": "+44",
    "Canada": "+1", "Australia": "+61", "Germany": "+49",
    "France": "+33", "Japan": "+81", "Brazil": "+55", "Mexico": "+52",
}

# ---------------- User Map ---------------- #
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

# ---------------- Face Detection & Recognition ---------------- #
def detect(frame, faceCascade, img_id, user_id, face_mesh, csv_writer):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    saved = False
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        os.makedirs("data", exist_ok=True)
        cv2.imwrite(f"data/user.{user_id}.{img_id}.jpg", face_img)
        saved = True

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=1)
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
            face_img = gray[y:y + h, x:x + w]
            id_, conf = clf.predict(face_img)
            if conf < threshold and str(id_) in user_map:
                user_info = user_map[str(id_)]
                name = user_info['name'] if isinstance(user_info, dict) else user_info
                label = f"{name}"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)
            center = (x + w // 2, y + h // 2)
            radius = int(max(w,h)/2)
            cv2.circle(frame, center, radius, color, 2)
            cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except:
            cv2.putText(frame, "Unknown", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=1)
            )
    return frame

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

# ---------------- Capture Function ---------------- #
def get_frame_cloud_or_local(user_id, max_images=50, cap=None):
    saved_count = 0
    stframe = st.empty()
    os.makedirs("landmarks", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    csv_file = open(f"landmarks/user_{user_id}.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while saved_count < max_images:
            if "IS_CLOUD" in os.environ:
                uploaded_image = st.camera_input("Take a picture")
                if uploaded_image is None:
                    st.info("Please take a picture...")
                    continue
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
            else:
                if cap is None:
                    cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break
            frame, saved = detect(frame, faceCascade, saved_count, user_id, face_mesh, csv_writer)
            if saved:
                saved_count += 1
            stframe.image(frame, channels="BGR", caption=f"Saved Images: {saved_count}/{max_images}")

    csv_file.close()
    if "IS_CLOUD" not in os.environ:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


# ---------------- Streamlit UI ---------------- #
st.set_page_config(page_title="Face Recognition", page_icon="🧑")
st.title("🧑 Real Time Face Recognition")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
user_map = load_user_map()
clf = None
if os.path.exists("classifier.yml"):
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.yml")

# ---------------- Sidebar ---------------- #
public_menu_options = ["📸 Capture Dataset","🧠 Train Model","🔍 Recognize Face","🖼️ Upload Image for Recognition"]
choice = st.sidebar.selectbox("Public Menu", public_menu_options)

st.sidebar.header("Admin Login")
with st.sidebar.form(key='login_form'):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submit_button = st.form_submit_button("Login")
    if submit_button:
        authenticate(username,password)

if st.session_state.authenticated:
    st.sidebar.subheader("Admin Actions")
    admin_choice = st.sidebar.selectbox("Admin Menu", ["📋 View Registered Users","🔄 Update All User Info","📊 View User Statistics","🗑️ Delete User","🚪 Logout"])
    choice = admin_choice

# ---------------- App Logic ---------------- #
# Capture Dataset
if choice=="📸 Capture Dataset":
    suggested_id = get_next_user_id()
    st.info(f"Next available User ID is {suggested_id}")
    user_id = st.number_input("Enter User ID", min_value=suggested_id, value=suggested_id, step=1)
    user_name = st.text_input("Enter User Name")
    selected_country = st.selectbox("Select Your Country", list(COUNTRY_CODES.keys()))
    country_code = COUNTRY_CODES[selected_country]
    user_phoneNo = st.text_input(f"Enter Your Phone No (e.g., {country_code} ...)")
    user_address = st.text_area("Enter Your Address")
    user_gender = st.selectbox("Select Your Gender", ["Male","Female","Other"])

    if st.button("Start Capture"):
        phone_without_code = user_phoneNo.strip()
        if not user_name.strip():
            st.error("⚠️ Enter a valid name!")
        elif not phone_without_code.isdigit() or len(phone_without_code)!=10:
            st.error("⚠️ Enter a valid 10-digit phone number!")
        else:
            full_phone_number = f"{country_code}{phone_without_code}"
            user_map[str(user_id)] = {"name":user_name,"phone":full_phone_number,"address":user_address,"gender":user_gender,"country":selected_country}
            save_user_map(user_map)
            cap = None if "IS_CLOUD" in os.environ else cv2.VideoCapture(0)
            get_frame_cloud_or_local(user_id, max_images=50, cap=cap)
            st.success(f"✅ Dataset captured for {user_name} (ID: {user_id})")

# Train Model
elif choice=="🧠 Train Model":
    if st.button("Train Now"):
        trained = train_classifier("data")
        if trained:
            clf = cv2.face.LBPHFaceRecognizer_create()
            clf.read("classifier.yml")
            st.success("✅ Model trained successfully!")
        else:
            st.error("❌ No valid data to train.")

# Recognize Face
elif choice=="🔍 Recognize Face":
    if clf is None or len(user_map)==0:
        st.warning("⚠️ No trained model or users. Capture dataset and train first.")
    elif st.button("Start Recognition"):
        st.info("🔎 Recognition started. Close Streamlit to stop.")
        cap = None if "IS_CLOUD" in os.environ else cv2.VideoCapture(0)
        stframe = st.empty()
        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            while True:
                if cap is not None:
                    ret, frame = cap.read()
                    if not ret: break
                else:
                    uploaded_image = st.camera_input("Take a picture")
                    if uploaded_image is None: continue
                    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, 1)
                frame = recognize(frame, clf, faceCascade, face_mesh, user_map)
                stframe.image(frame, channels="BGR")


elif choice=="🖼️ Upload Image for Recognition":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            frame = recognize(frame, clf, faceCascade, face_mesh, user_map)
        st.image(frame, channels="BGR")

elif choice=="📋 View Registered Users":
    if len(user_map)==0:
        st.info("No users registered.")
    else:
        data=[]
        for uid, info in user_map.items():
            if isinstance(info, dict):
                name = info.get("name","N/A")
                phone = info.get("phone","N/A")
                gender = info.get("gender","N/A")
                address = info.get("address","N/A")
                country = info.get("country","N/A")
            else:
                name=info; phone=gender=address=country="N/A"
            count = len([f for f in os.listdir("data") if f.startswith(f"user.{uid}.")])
            data.append({"User ID":uid,"Name":name,"Phone":phone,"Gender":gender,"Country":country,"Address":address,"Images":count})
        st.table(data)
        df = pd.DataFrame(data)
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Users")
        st.download_button("📥 Download Users as Excel", data=output.getvalue(), file_name="registered_users.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Delete User
elif choice=="🗑️ Delete User":
    del_id = st.number_input("Enter User ID to Delete", min_value=1, step=1)
    if st.button("Delete User"):
        if str(del_id) in user_map:
            for f in os.listdir("data"):
                if f.startswith(f"user.{del_id}."): os.remove(os.path.join("data",f))
            csv_path = f"landmarks/user_{del_id}.csv"
            if os.path.exists(csv_path): os.remove(csv_path)
            name = user_map[str(del_id)]["name"] if isinstance(user_map[str(del_id)], dict) else user_map[str(del_id)]
            user_map.pop(str(del_id))
            save_user_map(user_map)
            train_classifier("data")
            st.success(f"🗑️ Deleted user {name} (ID {del_id}) and retrained model.")
        else:
            st.error("User not found.")

# View User Statistics
elif choice=="📊 View User Statistics":
    total_users = len(user_map)
    total_images = len(os.listdir("data")) if os.path.exists("data") else 0
    st.metric("Total Registered Users", total_users)
    st.metric("Total Images in Dataset", total_images)
    if total_users>0:
        user_image_counts = {}
        for uid, info in user_map.items():
            name = info.get("name", f"User {uid}") if isinstance(info, dict) else info
            count = len([f for f in os.listdir("data") if f.startswith(f"user.{uid}.")])
            user_image_counts[name] = count
            st.write(f"**{name} (ID: {uid})** → {count} images")
        df = pd.DataFrame(list(user_image_counts.items()), columns=["User","Images"])
        fig, ax = plt.subplots()
        colors = cm.tab20(np.linspace(0,1,len(df)))
        ax.bar(df["User"], df["Images"], color=colors)
        ax.set_xlabel("Users"); ax.set_ylabel("Images"); ax.set_title("Images per User")
        st.pyplot(fig)
        fig2, ax2 = plt.subplots()
        ax2.pie(df["Images"], labels=df["User"], autopct="%1.1f%%", startangle=90)
        ax2.axis("equal"); ax2.set_title("Dataset Distribution")
        st.pyplot(fig2)

# Update All User Info
elif choice=="🔄 Update All User Info":
    st.subheader("Update User Information")
    current_users = {uid: info['name'] if isinstance(info, dict) else info for uid, info in user_map.items()}
    user_options = [f"{uid} - {name}" for uid,name in current_users.items()]
    selected_user = st.selectbox("Select User", options=["Select a User"] + user_options)
    if selected_user!="Select a User":
        upd_id = selected_user.split(" - ")[0]
        user_info = user_map.get(upd_id)
        if not isinstance(user_info, dict):
            user_info = {"name":user_info,"phone":"N/A","address":"N/A","gender":"N/A","country":"N/A"}
        phone_without_code = user_info.get("phone","N/A")
        if phone_without_code.startswith("+"):
            found=False
            for country, code in COUNTRY_CODES.items():
                if phone_without_code.startswith(code):
                    selected_country_default = country
                    phone_without_code = phone_without_code[len(code):]
                    found=True
                    break
            if not found: selected_country_default="India"
        else: selected_country_default=user_info.get("country","India")

        with st.form("update_form"):
            st.write(f"Updating user ID: **{upd_id}**")
            upd_name = st.text_input("User Name", value=user_info.get("name",""))
            upd_country = st.selectbox("Country", list(COUNTRY_CODES.keys()), index=list(COUNTRY_CODES.keys()).index(selected_country_default))
            upd_phoneNo = st.text_input("Phone No", value=phone_without_code)
            upd_gender = st.selectbox("Gender", ["Male","Female","Other"], index=["Male","Female","Other"].index(user_info.get("gender","Male")))
            upd_address = st.text_area("Address", value=user_info.get("address",""))
            submit_update_button = st.form_submit_button("Update User")
            if submit_update_button:
                updated_phone_without_code = upd_phoneNo.strip()
                if not upd_name.strip():
                    st.error("⚠️ User Name cannot be empty!")
                elif not updated_phone_without_code.isdigit() or len(updated_phone_without_code)!=10:
                    st.error("⚠️ Enter a valid 10-digit phone number!")
                else:
                    upd_full_phone = f"{COUNTRY_CODES[upd_country]}{updated_phone_without_code}"
                    user_map[upd_id] = {"name":upd_name,"phone":upd_full_phone,"gender":upd_gender,"country":upd_country,"address":upd_address}
                    save_user_map(user_map)
                    st.success(f"✅ User {upd_name} (ID {upd_id}) updated successfully!")

# Logout
elif choice=="🚪 Logout":
    logout()
