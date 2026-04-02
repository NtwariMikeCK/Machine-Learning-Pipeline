"""
gdrive_auth.py  —  run once to generate token.json
    python gdrive_auth.py
"""
from google_auth_oauthlib.flow import InstalledAppFlow
import json, os

SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDS_FILE = "credentials/oauth_client_secret.json"  # your downloaded OAuth JSON
TOKEN_FILE  = "credentials/token.json"

flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
creds = flow.run_local_server(port=0)

with open(TOKEN_FILE, "w") as f:
    f.write(creds.to_json())

print(f"✅ Token saved to {TOKEN_FILE}")