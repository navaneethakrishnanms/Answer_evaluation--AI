# -*- coding: utf-8 -*-
"""Script to remove all emojis from app_fixed.py"""

import re

# Read the file
with open('app_fixed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove all non-ASCII characters (emojis) and replace with safe alternatives
# This regex removes all Unicode emoji characters
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "]+", flags=re.UNICODE)

content = emoji_pattern.sub('[*]', content)

# Write back
with open('app_fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Emojis removed successfully!")
