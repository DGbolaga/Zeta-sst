import os
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv
from models import TranscriptionResponse
from fastapi import FastAPI, File, UploadFile, HTTPException
import assemblyai as aai

# Load environment variables (API_KEY) from .env file
load_dotenv()

API_KEY = os.getenv("API_KEY")

# Initialize AssemblyAI transcriber settings
aai.settings.api_key = API_KEY
transcriber = aai.Transcriber()

app = FastAPI()


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"message": "Transcription service is running. POST a file to /transcribe to begin."}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio_file(file: UploadFile = File(...)):
    """
    Receives a binary audio/video file, transcribes it, and returns the text.
    :param file: The uploaded audio or video file.
    """
    # 1. Set up temporary file path
    temp_dir = Path(tempfile.gettempdir())
    # Create a unique temporary filename based on the original name
    temp_file_path = temp_dir / f"{Path(file.filename).stem}_{os.urandom(8).hex()}{Path(file.filename).suffix}"
    
    print(f"Received file: {file.filename}. Saving temporarily to: {temp_file_path}")

    try:
        # 2. Save the uploaded file content to the temporary file
        # This is necessary because assemblyai.transcribe() expects a disk path or a public URL.
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Transcribe the file using the local path
        transcript = transcriber.transcribe(str(temp_file_path))

        # 4. Handle transcription status
        if transcript.status == aai.TranscriptStatus.error:
            error_message = f"Transcription failed: {transcript.error}. Check file format and content."
            print(error_message)
            raise HTTPException(status_code=400, detail=error_message)

        # 5. Return the result
        return TranscriptionResponse(transcript=transcript.text)

    except Exception as e:
        if not isinstance(e, HTTPException):
            print(f"An unexpected internal error occurred: {e}")
            raise HTTPException(status_code=500, detail=f"Internal Server Error during processing: {str(e)}")
        raise 

    finally:
        if temp_file_path.exists():
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")
