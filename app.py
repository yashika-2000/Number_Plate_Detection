import streamlit as st
import cv2
import numpy as np
import imutils
import easyocr

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Number Plate Recognition",
    page_icon="🚗",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
        .main {
            background-color: #0f172a;
        }
        h1, h2, h3 {
            color: #38bdf8;
        }
        .stButton>button {
            background-color: #38bdf8;
            color: black;
            font-weight: bold;
            border-radius: 10px;
            height: 3em;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🚗 Number Plate Detection & Recognition")
st.write("Upload a vehicle image to detect and read the number plate using **OpenCV & EasyOCR**.")

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "📤 Upload Vehicle Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- IMAGE RESIZE FUNCTION ----------------
def resize_image(image, max_width=700):
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

# ---------------- NUMBER PLATE DETECTION ----------------
def detect_number_plate(image):
    reader = easyocr.Reader(['en'], gpu=False)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    keypoints = cv2.findContours(
        edged.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is None:
        return image, "Number Plate Not Found"

    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [location], 0, 255, -1)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))

    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
    cropped_image = resize_image(cropped_image, max_width=400)

    result = reader.readtext(cropped_image)
    text = result[0][-2] if result else "Not Detected"

    cv2.rectangle(
        image,
        tuple(location[0][0]),
        tuple(location[2][0]),
        (0, 255, 0),
        3
    )

    cv2.putText(
        image,
        text,
        (location[0][0][0], location[0][0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return image, text

# ---------------- DISPLAY ----------------
if uploaded_file is not None:
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )
    image = cv2.imdecode(file_bytes, 1)
    image = resize_image(image, max_width=700)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            width=350
        )

    if st.button("🔍 Detect Number Plate"):
        result_img, plate_text = detect_number_plate(image.copy())

        with col2:
            st.subheader("✅ Processed Image")
            st.image(
                cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                width=350
            )

        st.success(f"🚘 **Detected Number Plate:** `{plate_text}`")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💡 Built with **Streamlit | OpenCV | EasyOCR**")
