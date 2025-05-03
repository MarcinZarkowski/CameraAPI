from flask import Flask
from sqlalchemy import create_engine, Column, Integer, Boolean, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import time
import cv2
import boto3
import openai
import os
from dotenv import load_dotenv
from threading import Lock, Thread
import datetime

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Flask app
app = Flask(__name__)

# DB setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

ROOM_NAME = "Daedalus Lounge"

class Room(Base):
    __tablename__ = "rooms"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    updating = Column(Boolean, default=False)
    description = Column(String, nullable=False)
    location = Column(String, nullable=False)
    last_updated = Column(DateTime(timezone=True), default=datetime.datetime.now, onupdate=datetime.datetime.now)
    current_occupancy = Column(Integer, nullable=True)
    total_occupancy = Column(Integer, nullable=False)
    computer_access = Column(Boolean, nullable=False)
    whiteboard_access = Column(Boolean, nullable=False)
    permitted_volume = Column(String, nullable=False)
    picture = Column(String, nullable=True)

Base.metadata.create_all(engine)

# AWS S3 setup
s3 = boto3.client('s3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

session = SessionLocal()
room = session.query(Room).filter(Room.name == ROOM_NAME).first()
if not room:
    room = Room(
        name="Daedalus Lounge",
        description="Lounge for Daedalus Honors students",
        location="Hunter North building, 10th floor.",
        total_occupancy=15,
        updating= True,
        computer_access=False,
        whiteboard_access=True,
        permitted_volume="There are no audio restrictions."
    )
    session.add(room)
    session.commit()
session.close()

BUCKET_NAME = os.getenv("S3_BUCKET")

# Thread-safe lock for loop execution
loop_lock = Lock()

def capture_image(filename="snapshot.jpg"):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
    cap.release()
    return filename if ret else None

def upload_to_s3(filepath, key):
    s3.upload_file(
        Filename=filepath,
        Bucket=BUCKET_NAME,
        Key=key
    )
    os.remove(filepath)
    return f"https://{BUCKET_NAME}.s3.amazonaws.com/{key}"


def analyze_image_via_gpt(image_url):
    # Build a multimodal user message containing both text and the image URL
    user_content = [
        {"type": "text",
         "text": "This is an image of a room. Return just a single number reflecting how many people are present."},
        {"type": "image_url",
         "image_url": {"url": image_url}}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",           # vision-enabled GPT-4
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant that classifies room occupancy. "
                    "Only return a single number – the count of people in the image."
                )
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        max_tokens=4
    )
    # Extract the assistant’s reply
    return response.choices[0].message.content.strip()



def monitor_loop():
    while True:
        with loop_lock:
            session = SessionLocal()
            room = session.query(Room).filter(Room.name == ROOM_NAME).first()
            if room and room.updating:
                print("Trigger detected! Taking photo...")
                filepath = capture_image()
                if filepath:
                    s3_url = upload_to_s3(filepath, f"room-{room.id}.jpg")
                    occupancy = analyze_image_via_gpt(s3_url)
                    room.current_occupancy = int(occupancy)
                    room.picture = s3_url
                    room.updating = False
                    session.commit()
                    print(f"Uploaded {filepath} to S3 and updated occupancy: {occupancy}")
                    print(s3_url)
            session.close()
           
        time.sleep(1)

# Start monitoring loop in a daemon thread
monitor_thread = Thread(target=monitor_loop, daemon=True)
monitor_thread.start()

if __name__ == '__main__':
    app.run(debug=True)
