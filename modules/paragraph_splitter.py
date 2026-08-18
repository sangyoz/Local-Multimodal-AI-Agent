# modules/paragraph_splitter.py
import re

def split_paragraphs(text, max_len=300):
    paragraphs = []
    current = ""

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if len(current) + len(line) < max_len:
            current += " " + line
        else:
            paragraphs.append(current.strip())
            current = line

    if current:
        paragraphs.append(current.strip())

    return paragraphs