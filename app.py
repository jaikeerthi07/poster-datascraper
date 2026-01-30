import streamlit as st
import cv2
import easyocr
import pandas as pd
import numpy as np
import re
import os
from tempfile import NamedTemporaryFile

# =============================
# STREAMLIT CONFIG
# =============================
st.set_page_config(
    page_title="Hackathon Poster Data Scraper",
    layout="wide"
)

st.title("🏆 Universal Hackathon Poster Data Scraper")
st.write("Upload **any hackathon poster** and automatically extract all details into a CSV file.")

# =============================
# LOAD OCR MODEL (CACHE)
# =============================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# =============================
# IMAGE PREPROCESSING
# =============================
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray

# =============================
# FIELD EXTRACTION LOGIC
# =============================
def extract_fields(lines):
    data = {
        "Event Name": "",
        "Organizer": "",
        "Prizes": [],
        "Important Dates": [],
        "Website": [],
        "Email": [],
        "Phone": [],
        "Other Info": []
    }

    for line in lines:
        text = line.strip()

        if re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text):
            data["Email"].append(text)

        elif re.search(r"(www\.|http)", text):
            data["Website"].append(text)

        elif re.fullmatch(r"\d{10}", text):
            data["Phone"].append(text)

        elif re.search(r"(₹|\$|Prize|Award|Cash)", text, re.IGNORECASE):
            data["Prizes"].append(text)

        elif re.search(r"\d{1,2}\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", text, re.IGNORECASE):
            data["Important Dates"].append(text)

        elif re.search(r"(Institute|University|Government|Dept|Foundation)", text, re.IGNORECASE):
            if not data["Organizer"]:
                data["Organizer"] = text

        elif text.isupper() and len(text) > 10:
            if not data["Event Name"]:
                data["Event Name"] = text

        else:
            data["Other Info"].append(text)

    return {
        "Event Name": data["Event Name"],
        "Organizer": data["Organizer"],
        "Prizes": " | ".join(data["Prizes"]),
        "Dates": " | ".join(data["Important Dates"]),
        "Website": " | ".join(data["Website"]),
        "Email": " | ".join(data["Email"]),
        "Phone": " | ".join(data["Phone"]),
        "Other Info": " | ".join(data["Other Info"])
    }

# =============================
# FILE UPLOAD
# =============================
uploaded_file = st.file_uploader(
    "📤 Upload Hackathon Poster Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    image = cv2.imread(image_path)
    st.image(image, caption="Uploaded Poster", use_column_width=True)

    with st.spinner("🔍 Extracting data using ML OCR..."):
        processed = preprocess_image(image)
        ocr_text = reader.readtext(processed, detail=0)
        structured_data = extract_fields(ocr_text)

    df = pd.DataFrame([structured_data])

    st.success("✅ Data extraction complete")

    st.subheader("📊 Extracted Data")
    st.dataframe(df, use_container_width=True)

    # =============================
    # DOWNLOAD CSV
    # =============================
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name="hackathon_data.csv",
        mime="text/csv"
    )
