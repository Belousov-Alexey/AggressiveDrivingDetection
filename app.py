from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import shutil
import uuid
import os

from main_pipeline import process_video

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
def upload_video(request: Request, file: UploadFile = File(...)):
    uid = str(uuid.uuid4())

    video_path = os.path.join(UPLOAD_DIR, f"{uid}.mp4")
    result_path = os.path.join(RESULT_DIR, uid)

    os.makedirs(result_path, exist_ok=True)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = process_video(video_path, result_path)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "video_url": f"/results/{uid}/{result['video']}",
            "report_url": f"/results/{uid}/{result['report']}",
            "total_tracks": result["total_tracks"],
            "aggressive_tracks": result["aggressive_tracks"],
            "total_events": result["total_events"],
            "input_video_name": result["input_video_name"]
        }
    )


app.mount("/results", StaticFiles(directory="results"), name="results")
