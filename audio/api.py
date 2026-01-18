import os, tempfile, json, requests, queue
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from pydub import AudioSegment
from proto_infer import classify_prototype
from twilio.request_validator import RequestValidator

# ----------------------------
# Config via env
# ----------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
VALIDATE_TWILIO = os.getenv("TWILIO_VALIDATE_SIGNATURE", "false").lower() == "true"

app = FastAPI()
cmd_queue = queue.Queue()

# Bird → (angle_deg, power_01, tap_pct). Tweak for your pygame.
BIRD_TO_PARAMS = {
    "red":   (35.0, 0.65, 0.45),
    "blue":  (30.0, 0.55, 0.25),
    "yellow":(25.0, 0.80, -1.0),
    "black": (40.0, 0.90, 0.95),
    "white": (50.0, 0.50, 0.60),
}

def convert_to_wav_mono16k_bytes(raw_bytes: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
        tmp_in.write(raw_bytes)
        in_path = tmp_in.name
    try:
        audio = AudioSegment.from_file(in_path)  # detects .ogg/.opus/.mp3/.wav
        audio = audio.set_channels(1).set_frame_rate(16000)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            out_path = tmp_out.name
        audio.export(out_path, format="wav")
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (locals().get("in_path"), locals().get("out_path")):
            if p:
                try: os.unlink(p)
                except: pass

def enqueue_fire(bird: str, confidence: float):
    angle, power, tap = BIRD_TO_PARAMS[bird]
    cmd_queue.put({
        "type":"fire",
        "bird": bird,
        "confidence": confidence,
        "angle": angle,
        "power": power,
        "tap": tap
    })

# -------- Local test: upload any audio, classify, enqueue
@app.post("/classify")
async def classify_upload(file: UploadFile = File(...)):
    raw = await file.read()
    wav_bytes = convert_to_wav_mono16k_bytes(raw)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_bytes); wav_path = tmp.name
    try:
        result = classify_prototype(wav_path)  # {'bird', 'confidence', 'probs'}
    finally:
        try: os.unlink(wav_path)
        except: pass
    enqueue_fire(result["bird"], result["confidence"])
    return JSONResponse(result)

# -------- Twilio WhatsApp webhook
@app.post("/twilio/whatsapp")
async def twilio_whatsapp(request: Request):
    form = await request.form()

    if VALIDATE_TWILIO:
        validator = RequestValidator(TWILIO_AUTH_TOKEN)
        sig = request.headers.get("X-Twilio-Signature", "")
        if not validator.validate(str(request.url), dict(form.items()), sig):
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    num_media = int(form.get("NumMedia", "0"))
    if num_media < 1:
        return PlainTextResponse('<Response><Message>Send a voice note.</Message></Response>',
                                 media_type="application/xml")

    media_url = form.get("MediaUrl0", "")
    if not media_url:
        return PlainTextResponse('<Response><Message>No media URL.</Message></Response>',
                                 media_type="application/xml")

    auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None
    r = requests.get(media_url, auth=auth, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Media download failed: {r.status_code}")

    wav_bytes = convert_to_wav_mono16k_bytes(r.content)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_bytes); wav_path = tmp.name
    try:
        result = classify_prototype(wav_path)
    finally:
        try: os.unlink(wav_path)
        except: pass

    enqueue_fire(result["bird"], result["confidence"])
    msg = f"Predicted bird: {result['bird']} (conf {result['confidence']:.2f})"
    return PlainTextResponse(f"<Response><Message>{msg}</Message></Response>",
                             media_type="application/xml")

# -------- pygame pulls next command (long-poll or short poll)
@app.get("/next-command")
def next_command():
    try:
        cmd = cmd_queue.get_nowayit=False
    except TypeError:
        # for older FastAPI/py versions; fallback to try/except
        pass

    try:
        cmd = cmd_queue.get_nowait()
        return JSONResponse(cmd)
    except queue.Empty:
        return JSONResponse({"type":"none"})
