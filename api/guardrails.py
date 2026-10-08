"""Guardrails appliqués au prompt utilisateur avant de l'envoyer au Service 4.

1. Validation Pydantic : nettoyage (caractères de contrôle, espaces) et longueur maximale.
2. Détection des tentatives d'injection de prompt (FR + EN) -> requête refusée.
3. Masquage des données personnelles (e-mail, téléphone) -> le LLM ne les voit jamais.
"""

import re
import unicodedata

from pydantic import BaseModel, Field, ValidationError, field_validator

from api.errors import GuardrailViolation

INJECTION_PATTERNS = [
    # Anglais
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules?|prompts?|guidelines?)\b",
    r"\byou\s+are\s+now\b",
    r"\b(act|pretend|behave)\s+as\b",
    r"\b(reveal|show|print|repeat)\b.{0,40}\b(system\s+prompt|instructions?)\b",
    r"\bjailbreak\b",
    r"\bDAN\s+mode\b",
    r"\bdeveloper\s+mode\b",
    # Français
    r"\b(ignore[rz]?|oublie[rz]?|contourne[rz]?)\b.{0,40}\b(instructions?|règles?|consignes?|prompts?)\b",
    r"\btu\s+es\s+(maintenant|désormais)\b",
    r"\b(fais|faites)\s+comme\s+si\b",
    r"\b(révèle|affiche|répète|montre)[rz]?\b.{0,40}\b(prompt\s+système|instructions?|consignes?)\b",
    r"\bmode\s+(développeur|sans\s+restriction)\b",
    # Balises de rôle propres aux LLM (ChatML, Llama, Markdown)
    r"<\|?\s*(im_start|im_end|system|endoftext)\s*\|?>",
    r"\[/?INST\]",
    r"^\s*#{2,}\s*(system|instruction)",
    r"^\s*(system|assistant)\s*:",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in INJECTION_PATTERNS]

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# Téléphones FR (06 12 34 56 78, 06.12.34.56.78, +33 6 12 34 56 78)
# et internationaux (+1 555 123 4567) : au moins 9 chiffres avec séparateurs optionnels.
PHONE_RE = re.compile(r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{1,4}\)?[\s.-]?){2,5}\d{2,4}(?!\w)")
_MIN_PHONE_DIGITS = 9

EMAIL_MASK = "[EMAIL]"
PHONE_MASK = "[TELEPHONE]"


class PromptInput(BaseModel):
    """Prompt utilisateur nettoyé. La longueur max est passée via le contexte de validation."""

    text: str = Field(min_length=1)

    @field_validator("text", mode="before")
    @classmethod
    def normalize(cls, v: str) -> str:
        v = unicodedata.normalize("NFKC", v)
        # Supprime les caractères de contrôle invisibles (sauf retours à la ligne et tabulations).
        v = "".join(c for c in v if c in "\n\t" or unicodedata.category(c)[0] != "C")
        v = re.sub(r"[ \t]+", " ", v)
        v = re.sub(r"\n{3,}", "\n\n", v)
        return v.strip()

    @field_validator("text")
    @classmethod
    def check_length(cls, v: str, info) -> str:
        max_chars = (info.context or {}).get("max_chars")
        if max_chars and len(v) > max_chars:
            raise ValueError(f"Prompt trop long ({len(v)} caractères, max {max_chars})")
        return v


def detect_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_RE)


def _mask_phone(match: re.Match) -> str:
    digits = sum(c.isdigit() for c in match.group())
    return PHONE_MASK if digits >= _MIN_PHONE_DIGITS else match.group()


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Remplace e-mails et téléphones par des marqueurs. Renvoie le texte et les types masqués."""
    found: list[str] = []
    text, n_email = EMAIL_RE.subn(EMAIL_MASK, text)
    if n_email:
        found.append("email")
    masked = PHONE_RE.sub(_mask_phone, text)
    if masked != text:
        found.append("telephone")
    return masked, found


def apply_guardrails(raw: str | None, max_chars: int) -> tuple[str | None, list[str]]:
    """Valide et nettoie le prompt. Renvoie (prompt_nettoyé, types_de_pii_masqués).

    Lève GuardrailViolation si le prompt est trop long ou ressemble à une injection.
    Un prompt absent ou vide renvoie None (le prompt reste optionnel).
    """
    if raw is None or not raw.strip():
        return None, []

    try:
        prompt = PromptInput.model_validate({"text": raw}, context={"max_chars": max_chars}).text
    except ValidationError as exc:
        reason = exc.errors()[0]["msg"].removeprefix("Value error, ")
        raise GuardrailViolation("invalid_prompt", reason)

    if detect_injection(prompt):
        raise GuardrailViolation(
            "prompt_injection",
            "Le prompt contient des instructions non autorisées (tentative d'injection détectée)",
        )

    return mask_pii(prompt)
