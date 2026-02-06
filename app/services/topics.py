# app/services/topics.py
import re
from typing import Dict, List, Set

# 1) Big-picture domain keywords (broad gate)
DOMAIN_KEYWORDS = [
    r"\bpain\b", r"\bchronic pain\b", r"\bnocicept", r"\bhyperalges", r"\ballodyn",
    r"\bopioid", r"\bopiates?\b", r"\bprescription opioid\b",
    r"\bsubstance use\b", r"\bSUD\b", r"\bmisuse\b", r"\bdependen(ce|t)\b",
    r"\balcohol\b", r"\bAUD\b", r"\bnicotine\b", r"\bsmok(ing|ers?)\b", r"\btobacco\b", r"\becig(ar?ette)?s?\b",
    r"\bcannabis\b", r"\bmarijuana\b",
    r"\banxiety\b", r"\bdepress(ion|ive)\b", r"\bPTSD\b", r"\bcatastrophiz", r"\bcrav(ing|e)\b",
    r"\bwithdraw(al)?\b", r"\brelapse\b", r"\bcessation\b", r"\bintervention\b", r"\btreatment\b",
    r"\bsleep\b", r"\binsomnia\b", r"\bfatigue\b"
]

# 2) Topic packs: each has a name and its terms (for filtering retrieved text)
TOPIC_PACKS: Dict[str, List[str]] = {
    "opioid": ["opioid", "opiates", "oxycodone", "hydrocodone", "morphine",
               "prescription opioid", "opioid misuse", "OUD", "opioid analgesic"],
    "nicotine": ["nicotine", "smoking", "cigarette", "e-cigarette", "ecig", "vaping", "tobacco"],
    "alcohol": ["alcohol", "AUD", "drinking", "binge", "AUDIT"],
    "cannabis": ["cannabis", "marijuana", "THC", "CB1", "CB2", "cannabinoid"],
    "pain_mechanisms": ["nociception", "hyperalgesia", "allodynia", "central sensitization",
                        "catastrophizing", "fear-avoidance"],
    "affect": ["anxiety", "depression", "negative affect", "distress", "anhedonia"],
    "sleep": ["sleep", "insomnia", "sleep disturbance", "sleep quality", "sleep efficiency"],
    "trauma_ptsd": ["PTSD", "posttraumatic", "trauma", "hyperarousal", "intrusion"],
    "cessation_relapse": ["cessation", "quit", "lapse", "relapse", "maintenance", "abstinence"],
}

_domain_re = re.compile("|".join(DOMAIN_KEYWORDS), re.IGNORECASE)

def is_domain_relevant(text: str) -> bool:
    return bool(_domain_re.search(text or ""))

def select_topic_terms(query: str) -> Set[str]:
    """Pick a union of packs based on query hints; fallback to empty (no extra filtering)."""
    q = (query or "").lower()
    chosen: Set[str] = set()

    # Simple rules; you can also map from intent classifier
    for pack, terms in TOPIC_PACKS.items():
        if any(t in q for t in terms):
            chosen.update(terms)

    # Fallback: if the query is very generic but clearly in-domain, don’t force a pack
    return chosen
