def run_python_code(code: str) -> str:
    import sys
    from io import StringIO
    stdout = StringIO()
    stderr = StringIO()
    sys.stdout = stdout
    sys.stderr = stderr
    try:
        exec(code, {})
    except SyntaxError as e:
        def get_string_version(code_str):
            return code_str.encode('utf-8').decode('unicode_escape')
        code = get_string_version(code)
        exec(code, {})
    return stdout.getvalue() + stderr.getvalue()