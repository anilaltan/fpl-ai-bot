"""
Simple diagnostics for Streamlit Auth config
Run: python diagnose_auth.py
This checks YAML structure, usernames, bcrypt-hash format, and cookie fields.
"""
import re
import yaml
from pathlib import Path

CFG = Path(__file__).parent / 'config.yaml'

HASH_RE = re.compile(r"^\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}$")


def mask(s: str, keep: int = 4) -> str:
    if not isinstance(s, str):
        return str(type(s))
    if len(s) <= keep:
        return '*' * len(s)
    return s[:keep] + '...' + s[-keep:]


def main():
    if not CFG.exists():
        print('❌ config.yaml not found at', CFG)
        return

    with open(CFG, 'r', encoding='utf-8') as f:
        try:
            cfg = yaml.safe_load(f)
        except Exception as e:
            print('❌ Failed to parse config.yaml:', e)
            return

    creds = cfg.get('credentials') if isinstance(cfg, dict) else None
    cookie = cfg.get('cookie') if isinstance(cfg, dict) else None

    if not creds or 'usernames' not in creds:
        print('❌ Missing credentials.usernames in config.yaml')
        return

    users = creds.get('usernames', {})
    print('Found users:', ', '.join(users.keys()) or '<none>')

    problems = False
    for u, info in users.items():
        pwd = info.get('password') if isinstance(info, dict) else None
        if not pwd or not isinstance(pwd, str):
            print(f" - {u}: ❌ missing or non-string password")
            problems = True
            continue
        if not HASH_RE.match(pwd):
            print(f" - {u}: ⚠️ password does not look like a standard bcrypt hash (length/format mismatch)")
            problems = True
        else:
            print(f" - {u}: ✅ bcrypt-like hash detected")

    if not cookie or not isinstance(cookie, dict):
        print('❌ Missing cookie config (cookie.name, cookie.key, cookie.expiry_days)')
        problems = True
    else:
        name = cookie.get('name')
        key = cookie.get('key')
        expiry = cookie.get('expiry_days')
        print('Cookie.name:', name or '<missing>')
        print('Cookie.key (masked):', mask(key or ''))
        print('Cookie.expiry_days:', expiry)
        if not key or not isinstance(key, str) or len(key) < 8:
            print('⚠️ cookie.key is missing or too short (should be a long random string)')
            problems = True

    if problems:
        print('\nAction items:')
        print(' - Use generate_password.py to create a fresh bcrypt hash and replace the password for a test user (e.g., admin).')
        print(' - Ensure each bcrypt hash is length 60 and starts with $2b$, $2a$ or $2y$.')
        print(' - Ensure cookie.key is a long random secret.')
        print(" - Run Streamlit with the app and check for any exception traces shown in the Streamlit logs.")
    else:
        print('\nNo obvious issues found in config.yaml. If login still returns None:')
        print(' - Confirm streamlit-authenticator version matches your code (run: pip show streamlit-authenticator).')
        print(' - Try regenerating one user password and test login.')


if __name__ == '__main__':
    main()
