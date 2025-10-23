from pydantic import BaseModel


# Define the structure for the successful response body
class TranscriptionResponse(BaseModel):
    """Schema for the successful transcription result."""
    transcript: str