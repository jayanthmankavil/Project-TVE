from flask import Flask, render_template, request, redirect, url_for
import cv2
import pytesseract
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
import os
import requests
from pytube import YouTube

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\jayan\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def download_video(url):
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    path = os.path.join(app.config['UPLOAD_FOLDER'], local_filename)
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192): 
            if chunk: 
                f.write(chunk)
    return path

def download_youtube_video(youtube_url):
    yt = YouTube(youtube_url)
    video = yt.streams.filter(file_extension='mp4').first()
    path = video.download(app.config['UPLOAD_FOLDER'])
    return path

def extract_text_from_frame(frame):
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang='eng', config='--oem 3 --psm 6')
    return text.strip()

def process_frame(frame):
    if frame is not None:
        return extract_text_from_frame(frame)
    return None

def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = frame_rate

    extracted_texts = []
    previous_text = ""

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(0, frame_count, frame_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                futures.append(executor.submit(process_frame, frame))

        for future in futures:
            text = future.result()
            if text and text != previous_text:
                extracted_texts.append(text)
                previous_text = text

    cap.release()
    return extracted_texts

@app.route('/', methods=['GET', 'POST'])
def index():
    extracted_texts = []
    if request.method == 'POST':
        video_file = request.files.get('file')
        video_url = request.form.get('url')
        youtube_url = request.form.get('youtube_url')

        if video_file:
            filename = secure_filename(video_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(filepath)
            extracted_texts = process_video(filepath)
        elif video_url:
            filepath = download_video(video_url)
            extracted_texts = process_video(filepath)
        elif youtube_url:
            filepath = download_youtube_video(youtube_url)
            extracted_texts = process_video(filepath)

    return render_template('index.html', extracted_texts=extracted_texts)

if __name__ == '__main__':
    app.run(debug=True)
