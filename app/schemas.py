from pydantic import BaseModel, Field, constr
from typing import Optional, Literal, List, Dict

MessageText = constr(strip_whitespace=True, min_length=1, max_length=8000)

class ChatRequest(BaseModel):
    thread_id: constr(strip_whitespace=True, min_length=1) = Field(..., description="Logical conversation id")
    message: MessageText = Field(..., description="User's raw message text")
    user_id: Optional[str] = Field(default=None, description="(Future) Authenticated user id")
    prefs: Optional[Dict[str, str]] = Field(default=None, description="(Optional) user preferences")

class IntentResult(BaseModel):
    intent: Literal["summarize","compare","extract","cite","critique","other"]
    confidence: float = Field(ge=0, le=1)

class SafetyResult(BaseModel):
    allowed: bool
    reason: Optional[str] = None
    replacement: Optional[str] = None

class ChatNormalized(BaseModel):
    thread_id: str
    message: str
    intent: IntentResult
    safety: SafetyResult
    normalized_message: str
    tags: List[str] = []
    context: Optional[List[Dict[str, str]]] = None   # 👈 add this

