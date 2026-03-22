import ast
import zipfile
import os
from pathlib import Path
from typing import Optional

# Tool vocabulary -- everything maps to this
TOOL_MAP = {
    # Environment access
    ('os', 'environ', 'get'):        'read_env',
    ('os', 'environ', '__getitem__'): 'read_env',
    ('os', 'getenv'):                'read_env',

    # File access
    ('open',):                       'read_file',
    ('pathlib', 'Path', 'read_text'): 'read_file',

    # Outbound network
    ('requests', 'post'):            'http_post',
    ('requests', 'get'):             'http_get',
    ('urllib', 'request', 'urlopen'): 'http_get',
    ('httpx', 'post'):               'http_post',
    ('httpx', 'get'):                'http_get',

    # Encoding / transformation
    ('base64', 'b64encode'):         'encode',
    ('base64', 'encodebytes'):       'encode',
    ('codecs', 'encode'):            'encode',

    # Shell execution
    ('subprocess', 'run'):           'exec_shell',
    ('subprocess', 'call'):          'exec_shell',
    ('subprocess', 'Popen'):         'exec_shell',
    ('os', 'system'):                'exec_shell',

    # Email
    ('smtplib', 'SMTP', 'sendmail'): 'send_email',

    # Socket
    ('socket', 'connect'):           'tcp_connect',
}

SENSITIVE_PATHS = {
    '.env', 'credentials', 'id_rsa', 'id_ed25519',
    '.aws', 'secrets', 'config', 'token', 'passwd',
    'shadow', 'keychain', 'vault'
}

SENSITIVE_ENV_KEYS = {
    'aws_secret', 'aws_access', 'api_key', 'api_secret',
    'password', 'passwd', 'secret', 'token', 'private_key',
    'auth', 'credential'
}


def resolve_call(node: ast.Call) -> Optional[str]:
    """Map an AST Call node to a tool label."""
    func = node.func

    # Simple call: requests.post(...)
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            key = (func.value.id, func.attr)
            if key in TOOL_MAP:
                return TOOL_MAP[key]
        # Chained: urllib.request.urlopen
        if isinstance(func.value, ast.Attribute):
            if isinstance(func.value.value, ast.Name):
                key = (func.value.value.id, func.value.attr, func.attr)
                if key in TOOL_MAP:
                    return TOOL_MAP[key]

    # Bare call: open(...)
    if isinstance(func, ast.Name):
        key = (func.id,)
        if key in TOOL_MAP:
            return TOOL_MAP[key]

    return None


def extract_arg(node: ast.Call) -> str:
    """Best-effort extraction of first argument as string."""
    if not node.args:
        return "unknown"
    arg = node.args[0]
    if isinstance(arg, ast.Constant):
        return str(arg.value)
    if isinstance(arg, ast.JoinedStr):  # f-string
        return "dynamic_string"
    return "computed"


def is_sensitive_arg(tool: str, arg: str) -> bool:
    """Check whether argument suggests sensitive data access."""
    arg_lower = arg.lower()
    if tool == 'read_env':
        return any(k in arg_lower for k in SENSITIVE_ENV_KEYS)
    if tool == 'read_file':
        return any(p in arg_lower for p in SENSITIVE_PATHS)
    return False


def extract_trace(source: str) -> list[dict]:
    """Parse Python source and return a list of tool call dicts."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    trace = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            tool = resolve_call(node)
            if tool:
                arg = extract_arg(node)
                trace.append({
                    'tool': tool,
                    'arg': arg,
                    'lineno': getattr(node, 'lineno', -1)
                })
    return trace


def is_exfiltration_trace(trace: list[dict]) -> bool:
    """
    Relaxed version: any env read OR sensitive file read
    combined with any outbound action.
    """
    has_read = any(
        s['tool'] in ('read_env', 'read_file')
        for s in trace
    )
    has_outbound = any(
        s['tool'] in ('http_post', 'http_get', 'exec_shell',
                      'send_email', 'tcp_connect')
        for s in trace
    )
    return has_read and has_outbound


def extract_from_zip(zip_path: str, password: bytes = b'infected') -> list[dict]:
    """
    Extract all Python files from a DataDog-format zip.
    Returns list of trace dicts.
    """
    results = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.setpassword(password)
            for name in zf.namelist():
                if name.endswith('.py'):
                    try:
                        source = zf.read(name).decode('utf-8', errors='ignore')
                        trace = extract_trace(source)
                        if trace:
                            results.append({
                                'file': name,
                                'trace': trace,
                                'is_exfiltration': is_exfiltration_trace(trace)
                            })
                    except Exception:
                        continue
    except Exception as e:
        pass
    return results
