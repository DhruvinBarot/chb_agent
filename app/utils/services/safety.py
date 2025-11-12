from app.schemas import SafetyResult

CLINICAL_TERMS = [
    "diagnose", "prescribe", "dosage", "dose",
    "increase my meds", "should i take", "is it safe if i"
]
SUICIDE_TERMS = [
    "suicidal", "kill myself", "end my life",
    "self-harm", "hurt myself"
]

def basic_safety_check(message: str) -> SafetyResult:
    text = message.lower()

    # Crisis/self-harm triage (non-clinical guidance only)
    if any(t in text for t in SUICIDE_TERMS):
        return SafetyResult(
            allowed=False,
            reason="crisis_content",
            replacement=(
                """I’m sorry you’re feeling this way. I can’t provide emergency help.
If you are in immediate danger, call your local emergency number now.
In the U.S., you can call or text 988 (Suicide & Crisis Lifeline)."""
            )
        )

    # Disallow medical advice requests (Step 1 scope)
    if any(t in text for t in CLINICAL_TERMS):
        return SafetyResult(
            allowed=False,
            reason="medical_advice",
            replacement=(
                """I can help discuss research evidence at a high level, but I can’t provide medical advice.
Consider discussing this with a licensed clinician. If you want a literature summary on a treatment, I can help."""
            )
        )

    return SafetyResult(allowed=True)
