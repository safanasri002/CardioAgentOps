from fastapi import UploadFile

from api.errors import InvalidUpload

ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
    "application/octet-stream",
}


def is_wav_header(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


async def read_and_validate_wav(file: UploadFile, max_mb: int) -> bytes:
    name = file.filename or ""
    if not name.lower().endswith(".wav"):
        raise InvalidUpload("Le fichier doit avoir l'extension .wav")
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise InvalidUpload(f"Content-type non supporté : {file.content_type}")

    limit = max_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise InvalidUpload(f"Fichier trop volumineux (max {max_mb} Mo)", 413)
    if not data:
        raise InvalidUpload("Fichier vide")
    if not is_wav_header(data):
        raise InvalidUpload("Contenu invalide : en-tête WAV (RIFF/WAVE) absent")
    return data
