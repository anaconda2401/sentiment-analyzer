import re
import pandas as pd

SYSTEM_PHRASES = [
    "end-to-end encrypted",
    "changed their phone number",
    "deleted this message",
    "security code changed",
    "added you",
    "left",
    "was added",
    "removed",
    "group link",
    "group icon",
    "group description",
]

MEDIA_PHRASES = ["<media omitted>", "<image omitted>", "<video omitted>", "<audio omitted>", "<sticker omitted>"]

# Covers Android and iOS WhatsApp export formats:
# Android: 27/08/2026, 8:02 PM - Sender: Message
# iOS:     [27/08/2026, 8:02:05 PM] Sender: Message
PATTERN = re.compile(
    r"^\[?(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})[,\s]+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[aApP][mM])?)\]?\s*[-\u2013]?\s*([^:]+):\s*(.*)$"
)
_TIMESTAMP_PREFIX = re.compile(r"^\[?\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}[,\s]+\d{1,2}:\d{2}")



def _is_system_message(message: str) -> bool:
    msg_lower = message.lower().strip()
    for phrase in SYSTEM_PHRASES:
        if phrase in msg_lower:
            return True
    return False


def _is_media_message(message: str) -> bool:
    msg_lower = message.lower().strip()
    for phrase in MEDIA_PHRASES:
        if phrase in msg_lower:
            return True
    return False


def parse_whatsapp_chat(filepath: str) -> pd.DataFrame:
    """
    Parses an exported WhatsApp .txt file into a DataFrame.
    Handles Android and iOS formats, multiline messages, reply/quote indicators,
    and filters system/media messages.
    Returns a DataFrame with columns: date, time, sender, message, is_reply, reply_preview.
    """
    data = []

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.rstrip('\n').rstrip('\r')
                if not line.strip():
                    continue

                match = PATTERN.match(line)
                if match:
                    date, time_str, sender, message = match.groups()

                    if _is_system_message(message):
                        continue
                    if _is_media_message(message):
                        continue

                    # Check for quote / reply markers (e.g. > quote or "..." quote)
                    is_reply = False
                    reply_preview = ""
                    trimmed_msg = message.strip()
                    if trimmed_msg.startswith(">") or trimmed_msg.startswith("“") or trimmed_msg.startswith('"'):
                        is_reply = True
                        # Extract first line as reply preview
                        lines = trimmed_msg.splitlines()
                        reply_preview = lines[0].lstrip('>').strip()[:60]

                    data.append({
                        'date': date.strip(),
                        'time': time_str.strip(),
                        'sender': sender.strip(),
                        'message': trimmed_msg,
                        'is_reply': is_reply,
                        'reply_preview': reply_preview
                    })
                else:
                    # Ignore standalone system/media notification lines
                    if _is_system_message(line) or _is_media_message(line) or _TIMESTAMP_PREFIX.match(line.strip()):
                        continue
                    # Continuation of a genuine multiline message
                    if data and line.strip():
                        # If continuation was a quote line
                        if line.strip().startswith(">"):
                            data[-1]['is_reply'] = True
                            if not data[-1]['reply_preview']:
                                data[-1]['reply_preview'] = line.strip().lstrip('>').strip()[:60]
                        data[-1]['message'] += ' ' + line.strip()
    except (IOError, OSError) as e:
        raise RuntimeError(f"Could not read chat file: {e}")

    if not data:
        return pd.DataFrame(columns=['date', 'time', 'sender', 'message', 'is_reply', 'reply_preview'])

    df = pd.DataFrame(data)
    # Drop rows where the message is empty after stripping
    df = df[df['message'].str.strip().str.len() > 0].reset_index(drop=True)
    return df


def extract_hour(time_str: str):
    """Safely extracts 24-hour value from a WhatsApp time string."""
    try:
        match = re.search(r'(\d{1,2}):(\d{2})', str(time_str))
        if match:
            hour = int(match.group(1))
            if 'pm' in str(time_str).lower() and hour != 12:
                hour += 12
            elif 'am' in str(time_str).lower() and hour == 12:
                hour = 0
            return hour
    except Exception:
        pass
    return None
