import re
import pandas as pd

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


def _regex_anonymize(text: str, name_map: dict) -> str:
    """Apply fast regex replacements and participant name substitution."""
    text = _URL_RE.sub('[URL]', text)
    text = _EMAIL_RE.sub('[EMAIL]', text)
    text = _PHONE_RE.sub('[PHONE]', text)
    # Replace participant names (longest first to avoid partial replacement)
    for real_name in sorted(name_map.keys(), key=len, reverse=True):
        if real_name in text:
            text = text.replace(real_name, name_map[real_name])
    return text


def _spacy_anonymize_batch(messages: list) -> list:
    """Use spaCy NER in batch mode to redact persons and locations not already tagged."""
    if _nlp is None:
        return messages

    result = []
    for doc in _nlp.pipe(messages, batch_size=256):
        text = doc.text
        for ent in reversed(doc.ents):
            # Skip tokens that are already tagged placeholders
            if _ALREADY_TAGGED.match(ent.text.strip()):
                continue
            if ent.label_ in ('GPE', 'LOC', 'FAC'):
                text = text[:ent.start_char] + '[ADDRESS]' + text[ent.end_char:]
            elif ent.label_ == 'PERSON':
                text = text[:ent.start_char] + '[PII]' + text[ent.end_char:]
        result.append(text)
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

        # Step 2 — fast regex pass on messages
        df_anon['message'] = df_anon['message'].apply(
            lambda txt: _regex_anonymize(txt, self._participant_map) if isinstance(txt, str) else txt
        )

        # Step 3 — spaCy NER batch pass for any remaining person/location names
        messages = df_anon['message'].tolist()
        df_anon['message'] = _spacy_anonymize_batch(messages)

        return df_anon
