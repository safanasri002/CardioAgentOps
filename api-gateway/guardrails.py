from pydantic import BaseModel, Field

# Mots interdits dans la réponse du LLM (diagnostic trop sûr, prescription...)
FORBIDDEN = ["certain", "100%", "aucun risque", "prescris", "posologie"]

DISCLAIMER = "Aide à la décision : ne remplace pas l'avis d'un médecin."


# Forme obligatoire du rapport renvoyé par les agents
class Report(BaseModel):
    label: str = Field(pattern="^(normal|abnormal)$")   # seulement ces 2 valeurs
    confidence: float = Field(ge=0, le=1)                # entre 0 et 1
    text: str = Field(max_length=2000)                   # texte du LLM


def check_input(filename: str, audio: bytes):
    """Guardrail d'entrée : renvoie un message d'erreur, ou None si tout va bien."""
    if not filename.endswith(".wav"):
        return "Envoie un fichier .wav"
    if not audio.startswith(b"RIFF"):          # un vrai .wav commence toujours par RIFF
        return "Le fichier n'est pas un vrai .wav"
    if len(audio) > 10 * 1024 * 1024:
        return "Fichier trop gros (max 10 Mo)"
    return None


def check_output(report: dict) -> dict:
    """Guardrail de sortie : valide et filtre la réponse du LLM."""
    r = Report(**report)                       # 1. erreur si la forme est mauvaise

    for word in FORBIDDEN:                     # 2. bloque les phrases dangereuses
        if word in r.text.lower():
            r.text = "Réponse bloquée par les guardrails. Consultez un cardiologue."
            break

    return {**r.model_dump(), "disclaimer": DISCLAIMER}   # 3. toujours un avertissement
