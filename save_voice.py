# MAKE SURE DOUBLE UNDERSCORES FOR --name-- and --main--  ###################

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import requests
import os

app = Flask(__name__)

# Twilio credentials


# Folder to save incoming media
MEDIA_FOLDER = "downloaded_media"
os.makedirs(MEDIA_FOLDER, exist_ok=True)

# Image to reply with
REPLY_IMAGE_URL = "https://i.pinimg.com/736x/b0/8a/d9/b08ad9219bc6c046a78c2f086ce9b058.jpg"

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    num_media = int(request.values.get("NumMedia", 0))
    response = MessagingResponse()

    if num_media > 0:
        media_url = request.values.get("MediaUrl0")
        media_type = request.values.get("MediaContentType0")
        
        # Determine extension from MIME type
        if "/" in media_type:
            extension = media_type.split("/")[-1]  # e.g., "jpeg", "ogg", "mp4", "pdf"
        else:
            extension = "bin"  # fallback
        
        # Download media with Twilio authentication
        file_data = requests.get(media_url, auth=(ACCOUNT_SID, AUTH_TOKEN)).content
        filename = os.path.join(
            MEDIA_FOLDER, f"media_{len(os.listdir(MEDIA_FOLDER)) + 1}.{extension}"
        )
        with open(filename, "wb") as f:
            f.write(file_data)
        
        msg = response.message(f"Thanks! Your media was saved as {filename}")
        msg.media(REPLY_IMAGE_URL)  # Reply with your image
    else:
        msg = response.message("Send us an image, video, audio, or document!")
    
    return str(response)

if __name__ == "__main__":
    app.run(port=5000)