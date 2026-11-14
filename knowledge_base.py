"""
knowledge_base.py - Clinical Knowledge Base and RAG Retrieval System for HealthCare AI
"""

from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Comprehensive, evidence-based medical knowledge documents
MEDICAL_KNOWLEDGE_DOCS = [
    {
        "id": "fever",
        "title": "Fever (Pyrexia) Guidance",
        "keywords": "fever high temperature chills shivering pyrexia body heat hot",
        "text": (
            "Fever (body temperature above 100.4°F or 38°C): Often an immune response to infection. "
            "Home Care: Rest, maintain hydration with water or electrolyte solutions, wear lightweight clothing, "
            "and take lukewarm sponge baths. Paracetamol or Ibuprofen may be used as directed by a physician. "
            "Red Flags: Fever exceeding 103°F (39.4°C) in adults, fever lasting over 3 consecutive days, "
            "stiff neck, confusion, seizure, rash, or persistent vomiting. Seek emergency care immediately if red flags appear."
        )
    },
    {
        "id": "cold",
        "title": "Common Cold (Viral Rhinitis)",
        "keywords": "cold runny nose sneezing nasal congestion blocked nose mild malaise",
        "text": (
            "Common Cold: A viral infection of the upper respiratory tract causing sneezing, runny or stuffy nose, "
            "mild fatigue, and low-grade discomfort. "
            "Home Care: Ample fluid intake, steam inhalation, saline nasal sprays, honey for soothing throat/cough, "
            "and plenty of sleep. Antibiotics do NOT cure viral colds. "
            "Red Flags: Shortness of breath, chest pain, high fever (>101°F), or symptoms worsening after 10-14 days. "
            "Consult a physician if symptoms do not improve."
        )
    },
    {
        "id": "headache",
        "title": "Headache (Cephalea) Guidance",
        "keywords": "headache migraine tension head pain throbbing temples forehead pressure",
        "text": (
            "Headache: Commonly caused by tension, stress, dehydration, lack of sleep, eye strain, or sinus pressure. "
            "Home Care: Hydrate with cool water, rest in a dark quiet room, apply cold or warm compresses to the forehead, "
            "and practice gentle neck stretches. "
            "Red Flags: 'Thunderclap' sudden severe explosive headache, headache accompanied by stiff neck, confusion, "
            "fever, numbness, visual aura/loss, or following a recent head trauma. Emergency care is critical for sudden severe headaches."
        )
    },
    {
        "id": "stomachache",
        "title": "Stomachache & Abdominal Pain",
        "keywords": "stomachache coldstomachache cold stomachache stomach pain abdominal pain cramps tummy ache gastritis indigestion nausea",
        "text": (
            "Stomachache & Abdominal Pain: Can arise from indigestion, gas, gastritis, gastroenteritis (stomach flu), "
            "or dietary irritation. "
            "Home Care: Sip clear liquids (water, broth, peppermint/ginger tea), follow the BRAT diet (bananas, rice, applesauce, toast), "
            "and avoid greasy, spicy, acidic foods or dairy. "
            "Red Flags: Severe rigid or localized pain (especially lower right quadrant indicating appendicitis), persistent vomiting, "
            "inability to keep liquids down, blood in stool/vomit, high fever, or pain radiating to the back/shoulder. Seek urgent care if severe."
        )
    },
    {
        "id": "cough",
        "title": "Cough (Acute & Persistent)",
        "keywords": "cough dry cough wet cough phlegm chest congestion hacking coughing",
        "text": (
            "Cough: A defensive reflex clearing airways; can be dry or productive (producing mucus), often triggered by "
            "viral infections, post-nasal drip, bronchitis, or environmental irritants. "
            "Home Care: Stay well hydrated, use a room humidifier or inhale steam, consume warm tea with honey (for adults/kids >1 yr), "
            "and elevate head during sleep. "
            "Red Flags: Coughing up blood (hemoptysis), struggling to breathe, persistent chest tightness, barking stridor cough, "
            "or a cough lasting more than 3 to 4 weeks. Consult a doctor promptly."
        )
    },
    {
        "id": "throat_pain",
        "title": "Throat Pain & Sore Throat (Pharyngitis)",
        "keywords": "throat pain sore throat painful swallowing scratchy throat pharyngitis swollen tonsils",
        "text": (
            "Throat Pain (Pharyngitis): Discomfort, pain, or scratchiness in the throat, often aggravated by swallowing. "
            "Common causes include viral colds, flu, and bacterial infections like Streptococcus (Strep throat). "
            "Home Care: Warm salt-water gargles (1/2 tsp salt in 1 cup warm water), warm herbal teas with honey, throat lozenges, "
            "and hydration. "
            "Red Flags: Difficulty swallowing saliva, inability to open mouth (trismus), breathing difficulty, rash, or high fever with white patches on tonsils. "
            "Requires medical evaluation and possible antibiotics if bacterial."
        )
    },
    {
        "id": "chest_pain",
        "title": "Chest Pain (Cardiac & Emergency Alert)",
        "keywords": "chest pain heart attack pressure tightness crushing chest heavy radiating arm jaw shortness of breath",
        "text": (
            "CRITICAL EMERGENCY ALERT - Chest Pain: Chest discomfort or pain can be an indicator of life-threatening cardiac events "
            "such as myocardial infarction (heart attack), angina, pulmonary embolism, or aortic dissection. "
            "Immediate Action: Treat new, unexplained, pressing, squeezing, or crushing chest pain as a medical emergency. "
            "Dial emergency services (911 / 112 / 999) or proceed immediately to the nearest Emergency Room. "
            "Red Flags: Pain radiating to the left arm, neck, jaw, back, or stomach, accompanied by shortness of breath, cold sweat, dizziness, or nausea. "
            "Do NOT delay or attempt self-treatment."
        )
    },
    {
        "id": "tooth_pain",
        "title": "Tooth Pain & Dental Emergencies",
        "keywords": "tooth pain toothache dental ache swollen gum cavity molar sensitive tooth",
        "text": (
            "Tooth Pain (Odontalgia): Typically caused by tooth decay, cavities, damaged fillings, dental abscess, "
            "gum disease, or exposed tooth root. "
            "Home Care: Rinse mouth with warm salt water, gently floss to remove trapped food, apply a cold pack to the cheek, "
            "and use over-the-counter pain relievers temporarily. Avoid placing aspirin directly on the gums. "
            "Red Flags: Swelling of the face, cheek, or neck lymph nodes, fever, foul-tasting drainage/pus, or difficulty swallowing. "
            "Requires direct dental examination; dental infections do not resolve without professional dental intervention."
        )
    },
    {
        "id": "diabetes",
        "title": "Diabetes Mellitus Care & Blood Sugar Management",
        "keywords": "diabetes diabetis blood sugar glucose insulin high sugar polyuria polydipsia thirst frequent urination",
        "text": (
            "Diabetes Mellitus: A metabolic condition characterized by chronic hyperglycemia resulting from defects in insulin secretion, "
            "action, or both (Type 1, Type 2, or Gestational). "
            "Key Principles: Regular self-monitoring of blood glucose, compliance with prescribed insulin or oral hypoglycemics, "
            "balanced low-glycemic nutrition, regular physical activity, and foot inspections. "
            "Red Flags: Severe hypoglycemia (shakiness, confusion, sweating, loss of consciousness - treat immediately with 15g fast-acting carbs) "
            "or severe hyperglycemia with fruity breath, rapid breathing, and vomiting (Diabetic Ketoacidosis / DKA - requires immediate hospital care)."
        )
    },
    {
        "id": "asthma",
        "title": "Asthma & Bronchial Airway Management",
        "keywords": "asthma wheezing inhaler shortness of breath breathing difficulty bronchospasm asthma attack",
        "text": (
            "Asthma: A chronic inflammatory condition of the airways causing recurring episodes of wheezing, breathlessness, "
            "chest tightness, and coughing, often worse at night or early morning. "
            "Management: Follow an Asthma Action Plan, use prescribed preventer/controller inhalers daily, and carry a quick-relief "
            "bronchodilator (such as Albuterol/Salbutamol) for sudden attacks. Avoid personal triggers like smoke, pollen, or cold air. "
            "Red Flags (Asthma Emergency): Severe breathlessness where speaking in full sentences is impossible, blue lips or nails, "
            "retractions (ribs/neck pulling inward during breathing), or no relief from rescue inhaler after multiple puffs. Call emergency services immediately."
        )
    }
]

# Normalization mapping for common colloquialisms or typos
TYPO_MAPPING = {
    "coldstomachache": "cold stomachache stomach pain abdominal pain",
    "diabetis": "diabetes blood glucose",
    "stomach ache": "stomachache abdominal pain",
    "head ache": "headache",
    "tooth ache": "toothache tooth pain",
}

class HealthcareRAG:
    """
    TF-IDF Vectorized Retrieval-Augmented Generation (RAG) Engine
    with cosine similarity ranking and query preprocessing.
    """
    def __init__(self, docs: List[Dict] = None):
        self.docs = docs if docs is not None else MEDICAL_KNOWLEDGE_DOCS
        self.corpus = [
            f"{d['title']}. {d['keywords']}. {d['text']}"
            for d in self.docs
        ]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)  # Capture single words and bi-grams like "chest pain", "throat pain"
        )
        self.doc_vectors = self.vectorizer.fit_transform(self.corpus)

    def preprocess_query(self, query: str) -> str:
        """Preprocesses user query, normalizing common typos and concatenated words."""
        cleaned = query.lower()
        for typo, fix in TYPO_MAPPING.items():
            if typo in cleaned:
                cleaned = cleaned.replace(typo, fix)
        return cleaned

    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[Dict, float]]:
        """
        Retrieves the top_k most relevant medical knowledge documents
        along with their cosine similarity scores.
        """
        processed_q = self.preprocess_query(query)
        q_vec = self.vectorizer.transform([processed_q])
        similarities = cosine_similarity(q_vec, self.doc_vectors)[0]

        # Rank documents by cosine similarity descending
        ranked_indices = similarities.argsort()[::-1]

        results = []
        for idx in ranked_indices[:top_k]:
            score = float(similarities[idx])
            results.append((self.docs[idx], round(score, 4)))

        return results

    def get_best_context(self, query: str) -> Tuple[str, str, float]:
        """
        Returns (document_title, document_text, score) for the top match.
        """
        top_matches = self.retrieve(query, top_k=1)
        if top_matches:
            doc, score = top_matches[0]
            return doc["title"], doc["text"], score
        return "General Medical Context", "Consult a certified healthcare provider.", 0.0
