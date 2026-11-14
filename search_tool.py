"""
search_tool.py - External Live Web Search Tool for HealthCare AI
Integrates DuckDuckGo Search querying verified medical sources (NHS, MedlinePlus, CDC, Mayo Clinic).
"""

from typing import List, Dict
import time

# Robust import for DuckDuckGo
try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None

# Reliable fallback clinical summaries in case network or DDGS rate limits block live lookup
CURATED_WEB_FALLBACKS = {
    "fever": (
        "NHS & MedlinePlus: A fever is a common sign of illness and usually resolves in a few days. "
        "Keep the room well-ventilated, drink plenty of clear fluids, and contact a GP if fever persists beyond 3 days "
        "or exceeds 39.4°C."
    ),
    "cold": (
        "MedlinePlus & NHS: Most common colds last 7 to 10 days. Treatments focus on relieving symptoms: "
        "rest, staying hydrated, humidifiers, and saline gargles. Antibiotics do not work against viruses."
    ),
    "headache": (
        "Mayo Clinic & NHS: Most headaches are primary (tension or migraine) and manageable with rest and hydration. "
        "Sudden, excruciating 'worst headache of life' or headaches with neurological symptoms require emergency evaluation."
    ),
    "stomachache": (
        "NHS & MedlinePlus: Abdominal pain can stem from indigestion, gas, or gastroenteritis. "
        "Sip fluids slowly. Seek immediate care if accompanied by high fever, repeated vomiting, or pain localized to the lower right."
    ),
    "cough": (
        "CDC & NHS: Acute coughs are typically viral. Honey (for people >1 year) and fluids provide relief. "
        "A cough lasting over 3 weeks, coughing blood, or accompanied by breathing difficulty warrants prompt physician examination."
    ),
    "throat pain": (
        "NHS: Sore throats are mostly caused by viral infections. Salt water gargles and warm liquids relieve pain. "
        "If you have high fever, severe difficulty swallowing, or swollen glands, see a doctor to check for Strep throat."
    ),
    "chest pain": (
        "EMERGENCY ADVICE - NHS & American Heart Association: Unexplained chest pain, tightness, or pressure is a medical emergency. "
        "Call 911 / 999 or emergency services immediately. Do not drive yourself to the hospital."
    ),
    "tooth pain": (
        "NHS & ADA: Toothaches usually stem from tooth decay or pulpitis. Over-the-counter analgesics offer temporary relief, "
        "but professional dental assessment is required to treat the root cause."
    ),
    "diabetes": (
        "CDC & WHO: Diabetes is managed through lifestyle measures, regular blood glucose tracking, medication adherence, "
        "and routine check-ups. Recognize early symptoms of hypoglycemia and hyperglycemia."
    ),
    "asthma": (
        "Global Initiative for Asthma (GINA) & NHS: Asthma requires an ongoing action plan with controller and rescue inhalers. "
        "Severe attacks that do not respond to a rescue inhaler require emergency hospital treatment."
    )
}

def query_medical_web(query: str, max_results: int = 2) -> Dict:
    """
    Queries live web search via DuckDuckGo for clinical health information
    restricted to verified medical domains (NHS, MedlinePlus, CDC, WHO).

    Returns a dict with 'snippets', 'source', and 'status'.
    """
    clean_q = query.strip()

    # Formulate search query restricted to trusted medical sources
    search_query = f"{clean_q} site:nhs.uk OR site:medlineplus.gov OR site:cdc.gov"

    results = []
    status = "live"

    if DDGS is not None:
        try:
            with DDGS() as ddgs:
                ddg_gen = ddgs.text(search_query, max_results=max_results)
                for item in ddg_gen:
                    snippet = item.get("body", "")
                    title = item.get("title", "")
                    href = item.get("href", "")
                    if snippet:
                        results.append(f"[{title}] {snippet} (Source: {href})")
        except Exception as e:
            status = f"ddgs_exception: {str(e)}"

    # If live search returned empty or encountered rate-limiting/network error, use curated clinical fallback
    if not results:
        status = "curated_fallback"
        matched_fallback = None
        lower_q = clean_q.lower()
        for key, text in CURATED_WEB_FALLBACKS.items():
            if key in lower_q or (key == "stomachache" and "coldstomachache" in lower_q) or (key == "diabetes" and "diabetis" in lower_q):
                matched_fallback = text
                break

        if not matched_fallback:
            matched_fallback = (
                "MedlinePlus & NHS Health Guidance: Consult a registered medical doctor or certified healthcare practitioner "
                "for personalized medical diagnosis, prescription, and triage."
            )
        results.append(matched_fallback)

    return {
        "status": status,
        "query_used": search_query,
        "snippets": results,
        "combined_text": "\n\n".join(results)
    }
