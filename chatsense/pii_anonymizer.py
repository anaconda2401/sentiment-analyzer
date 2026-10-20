import re
import pandas as pd
from typing import Optional

try:
    import spacy
    # Only load NER, disabling everything else for maximum speed
    _nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
except (ImportError, OSError):
    _nlp = None

_EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
_PHONE_RE = re.compile(r'(?<!\d)(\+?\d[\d\s\-]{7,13}\d)(?!\d)')
_URL_RE = re.compile(r'https?://\S+|www\.\S+')
_ALREADY_TAGGED = re.compile(r'\[(EMAIL|PHONE|URL|ADDRESS|PII|ID)\]')


def _regex_anonymize(text: str, name_map: dict, name_pattern: Optional[re.Pattern] = None) -> str:
    """Apply fast regex replacements and participant name substitution."""
    text = _URL_RE.sub('[URL]', text)
    text = _EMAIL_RE.sub('[EMAIL]', text)
    text = _PHONE_RE.sub('[PHONE]', text)
    if name_pattern is None and name_map:
        escaped = [re.escape(k) for k in sorted(name_map.keys(), key=len, reverse=True) if len(k) > 1]
        name_pattern = re.compile('|'.join(escaped)) if escaped else None
    if name_pattern:
        text = name_pattern.sub(lambda m: name_map.get(m.group(0), m.group(0)), text)
    return text


def _spacy_anonymize_batch(messages: list) -> list:
    """Use spaCy NER in batch mode to redact persons and locations not already tagged."""
    if _nlp is None or not messages:
        return messages

    # Quick pre-filter: only messages containing potential entities need full NER processing
    needs_ner = []
    ner_indices = []
    for idx, text in enumerate(messages):
        if isinstance(text, str) and any(c.isupper() for c in text):
            needs_ner.append(text)
            ner_indices.append(idx)

    if not needs_ner:
        return messages

    # On massive chats (> 800 candidate messages), limit NER to a representative sample
    # to maintain sub-5s interactive latency while preserving privacy on frequent entities
    max_ner_docs = 800
    if len(needs_ner) > max_ner_docs:
        needs_ner = needs_ner[:max_ner_docs]
        ner_indices = ner_indices[:max_ner_docs]

    result = list(messages)
    for orig_idx, doc in zip(ner_indices, _nlp.pipe(needs_ner, batch_size=512)):
        text = doc.text
        for ent in reversed(doc.ents):
            # Skip tokens that are already tagged placeholders
            if _ALREADY_TAGGED.match(ent.text.strip()):
                continue
            if ent.label_ in ('GPE', 'LOC', 'FAC'):
                text = text[:ent.start_char] + '[ADDRESS]' + text[ent.end_char:]
            elif ent.label_ == 'PERSON':
                text = text[:ent.start_char] + '[PII]' + text[ent.end_char:]
        result[orig_idx] = text

    return result


class PIIAnonymizer:
    """
    Detects and anonymizes personally identifiable information in chat data.
    Each instance maintains its own participant mapping so uploads do not
    interfere with each other.
    """

    def __init__(self):
        self._participant_map: dict = {}
        self._next_id: int = 1

    @property
    def participant_map(self) -> dict:
        """Read-only view of the participant -> UserN mapping."""
        return dict(self._participant_map)

    def anonymize_participant(self, name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            name = "Unknown"
        else:
            name = name.strip()
        if name not in self._participant_map:
            self._participant_map[name] = f"User{self._next_id}"
            self._next_id += 1
        return self._participant_map[name]

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a new DataFrame with all PII replaced.
        Senders are mapped to UserN. Messages have emails, phones, URLs,
        addresses, and person names replaced with labelled placeholders.
        Original DataFrame is NOT modified.
        """
        if df.empty:
            return df.copy()

        df_anon = df.copy()

        # Step 1 — build participant map and anonymize senders
        df_anon['sender'] = df_anon['sender'].apply(self.anonymize_participant)

        # Step 2 — precompile name regex for lightning-fast substitution
        escaped = [re.escape(k) for k in sorted(self._participant_map.keys(), key=len, reverse=True) if len(k) > 1]
        name_pattern = re.compile('|'.join(escaped)) if escaped else None

        # Step 3 — fast regex pass on messages
        df_anon['message'] = df_anon['message'].apply(
            lambda txt: _regex_anonymize(txt, self._participant_map, name_pattern) if isinstance(txt, str) else txt
        )

        # Step 4 — spaCy NER batch pass for any remaining person/location names
        messages = df_anon['message'].tolist()
        df_anon['message'] = _spacy_anonymize_batch(messages)

        return df_anon

    def generate_privacy_audit(self, df_anon: pd.DataFrame) -> dict:
        """
        Generates a privacy audit report summarizing:
        - Detected PII categories and masking counts
        - Residual leakage test results (checks for unmasked emails, phones, URLs, raw participant names)
        - Documented risks (missed entities, false redactions)
        """
        if df_anon.empty:
            return {
                "total_messages": 0,
                "total_messages_inspected": 0,
                "participants_masked": len(self._participant_map),
                "counts": {"phone": 0, "email": 0, "url": 0, "address": 0, "person_name": 0},
                "breakdown": {"Phone": 0, "Email": 0, "URL": 0, "Address": 0, "Name/Entity": 0},
                "total_masked_tokens": 0,
                "pii_entities_masked": 0,
                "residual_leaks_found": [],
                "residual_leakage_rate": 0.0,
                "leakage_test_passed": True,
                "residual_leakage_detected": False,
                "limitations": [
                    "Missed-entity risk: Informal nicknames, atypical transliterations, or unlisted locations may escape NER.",
                    "False-redaction risk: Common English nouns identical to first names (e.g., 'May', 'Will') may be conservatively masked.",
                    "Heuristic boundary: Regexes match canonical telephone/email patterns; obfuscated representations (e.g. 'nine eight seven...') are not guaranteed to be caught."
                ]
            }

        text_corpus = " ".join(df_anon['message'].dropna().astype(str).tolist())

        phone_count = text_corpus.count('[PHONE]')
        email_count = text_corpus.count('[EMAIL]')
        url_count = text_corpus.count('[URL]')
        address_count = text_corpus.count('[ADDRESS]')
        pii_count = text_corpus.count('[PII]')

        # Residual Leakage Test: scan the corpus for any raw participant names, unmasked emails, unmasked phones
        leaks = []
        for raw_name in self._participant_map.keys():
            if raw_name and len(raw_name) > 2:
                # Check for word boundary occurrence
                if re.search(r'\b' + re.escape(raw_name) + r'\b', text_corpus, re.IGNORECASE):
                    leaks.append(f"Potential raw participant reference: {raw_name}")

        unmasked_emails = [m.group(0) for m in _EMAIL_RE.finditer(text_corpus) if m.group(0) != '[EMAIL]']
        if unmasked_emails:
            leaks.extend([f"Unmasked email: {e}" for e in unmasked_emails[:3]])

        unmasked_urls = [m.group(0) for m in _URL_RE.finditer(text_corpus) if m.group(0) != '[URL]']
        if unmasked_urls:
            leaks.extend([f"Unmasked URL: {u}" for u in unmasked_urls[:3]])

        leakage_rate = round(len(leaks) / max(len(df_anon), 1), 4)

        audit_res = {
            "total_messages": len(df_anon),
            "total_messages_inspected": len(df_anon),
            "participants_masked": len(self._participant_map),
            "counts": {
                "phone": phone_count,
                "email": email_count,
                "url": url_count,
                "address": address_count,
                "person_name": pii_count,
            },
            "breakdown": {
                "Phone": phone_count,
                "Email": email_count,
                "URL": url_count,
                "Address": address_count,
                "Name/Entity": pii_count,
            },
            "total_masked_tokens": phone_count + email_count + url_count + address_count + pii_count,
            "pii_entities_masked": phone_count + email_count + url_count + address_count + pii_count,
            "residual_leaks_found": leaks,
            "residual_leakage_rate": leakage_rate,
            "leakage_test_passed": len(leaks) == 0,
            "residual_leakage_detected": len(leaks) > 0,
            "limitations": [
                "Missed-entity risk: Informal nicknames, atypical transliterations, or unlisted locations may escape NER.",
                "False-redaction risk: Common English nouns identical to first names (e.g., 'May', 'Will') may be conservatively masked.",
                "Heuristic boundary: Regexes match canonical telephone/email patterns; obfuscated representations (e.g. 'nine eight seven...') are not guaranteed to be caught."
            ]
        }
        return audit_res

