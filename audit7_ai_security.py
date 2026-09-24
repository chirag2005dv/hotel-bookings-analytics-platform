import re
import os
from pathlib import Path

print('=== AUDIT 7: AI INTEGRATION SECURITY ===')

ai_src  = Path('ai_insights.py').read_text(encoding='utf-8')
env_src = Path('.env.example').read_text(encoding='utf-8')

# 7a. No hardcoded API key in ai_insights.py
key_patterns = [
    r'sk-[A-Za-z0-9]{20,}',
    r'api_key\s*=\s*["\'][^"\']{10,}["\']',
    r'OPENAI_API_KEY\s*=\s*["\'][^"\']+["\']',
]
for pat in key_patterns:
    matches = re.findall(pat, ai_src)
    assert not matches, f'Possible hardcoded key in ai_insights.py: {matches}'
print('  ai_insights.py: no hardcoded API keys  OK')

# 7b. Key is read ONLY from environment
assert 'os.environ.get("OPENAI_API_KEY"' in ai_src, \
    'API key must be read from os.environ'
assert 'load_dotenv(' in ai_src, \
    'python-dotenv must be used to load .env'
print('  ai_insights.py: key read from os.environ + load_dotenv()  OK')

# 7c. .env.example exists and contains placeholder (not a real key)
assert '.env.example' and Path('.env.example').exists()
assert 'sk-...your-key-here...' in env_src or 'your-key' in env_src, \
    '.env.example should contain a placeholder, not a real key'
# Ensure no real key pattern in .env.example
for pat in key_patterns[:1]:  # only the sk- pattern
    m = re.findall(pat, env_src)
    assert not m, f'Real key found in .env.example: {m}'
print('  .env.example: contains placeholder only, no real key  OK')

# 7d. No .env file (should not be committed)
dot_env = Path('.env')
if dot_env.exists():
    env_content = dot_env.read_text(encoding='utf-8', errors='ignore')
    real_keys = re.findall(r'sk-[A-Za-z0-9]{20,}', env_content)
    assert not real_keys, f'.env contains a real API key — do not commit this file'
    print('  .env: exists but contains no real key pattern  OK')
else:
    print('  .env: does not exist (good for repo safety)  OK')

# 7e. Fallback message is present and safe (no key disclosure)
from ai_insights import FALLBACK_MSG
assert 'OPENAI_API_KEY' in FALLBACK_MSG
assert 'sk-' not in FALLBACK_MSG
print('  FALLBACK_MSG: references env var name, no key value  OK')

# 7f. Grounding instruction forbids fabrication
from ai_insights import GROUNDING_INSTRUCTION
required_phrases = [
    'ONLY',
    'Do NOT',
    'not present in the data',
]
grounding_upper = GROUNDING_INSTRUCTION.upper()
for phrase in required_phrases:
    assert phrase.upper() in grounding_upper, \
        f'Grounding instruction missing required phrase: {phrase}'
print('  GROUNDING_INSTRUCTION: all anti-fabrication clauses present  OK')

# 7g. Temperature is set low (≤ 0.3) for factual outputs
temp_match = re.search(r'temperature\s*=\s*([0-9.]+)', ai_src)
assert temp_match, 'temperature parameter not found'
temp = float(temp_match.group(1))
assert temp <= 0.3, f'Temperature {temp} too high for factual outputs (should be ≤ 0.3)'
print(f'  API temperature: {temp} (<= 0.3 for factual consistency)  OK')

# 7h. Timeout is explicitly set
timeout_match = re.search(r'DEFAULT_TIMEOUT\s*=\s*(\d+)', ai_src)
assert timeout_match, 'DEFAULT_TIMEOUT not found in ai_insights.py'
timeout_val = int(timeout_match.group(1))
assert 10 <= timeout_val <= 120, f'Timeout {timeout_val}s outside expected range'
print(f'  DEFAULT_TIMEOUT: {timeout_val}s  OK')

# 7i. Retry logic is present
assert 'MAX_RETRIES' in ai_src
retry_match = re.search(r'MAX_RETRIES\s*=\s*(\d+)', ai_src)
retries = int(retry_match.group(1))
assert retries >= 1, 'At least 1 retry should be configured'
print(f'  MAX_RETRIES: {retries}  OK')

# 7j. AuthenticationError handled explicitly (no key leakage in error message)
assert 'AuthenticationError' in ai_src
# Ensure the auth error handler does NOT echo the api_key value
auth_handler_match = re.search(
    r'AuthenticationError.*?return\s+["\'](.+?)["\']',
    ai_src, re.DOTALL
)
if auth_handler_match:
    handler_text = auth_handler_match.group(1)
    assert 'sk-' not in handler_text, 'Auth error handler must not echo the key'
print('  AuthenticationError: handled without key disclosure  OK')

# 7k. Prompt builders only reference pre-loaded validated data (no raw df access)
# Verify none of the build_* functions read hotel_bookings_cleaned.csv directly
import ast
tree = ast.parse(ai_src)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name.startswith('build_'):
        func_src = ast.unparse(node)
        assert 'read_csv' not in func_src, \
            f'{node.name} calls read_csv — prompts must use pre-loaded data only'
print('  Prompt builders: no direct CSV reads (use pre-loaded validated dicts)  OK')

# 7l. Syntax check
import py_compile
py_compile.compile('ai_insights.py', doraise=True)
print('  ai_insights.py: syntax valid  OK')

print('\nAUDIT 7 PASSED')
