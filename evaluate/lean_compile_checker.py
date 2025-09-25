import subprocess
import tempfile
import json
import os

def validate_lean_code(code: str) -> str:
    """
    Validate whether the provided Lean 4 code passes compilation.
    
    Args:
        code (str): Lean code as a string.
    
    Returns:
        str: "pass" if compilation succeeds, otherwise an error or warning message.
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False, encoding='utf-8') as f:
            temp_path = f.name
            f.write(code)
            f.flush()

        result = subprocess.run(
            ["lean", temp_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=60
        )

        if result.returncode != 0:
            full_output = (result.stdout + "\n" + result.stderr).strip()
            error_lines = [
                line for line in full_output.split('\n')
                if "error" in line.lower() or "warning" in line.lower()
            ]
            error_msg = "\n".join(error_lines) or "Unknown compilation error"
            error_msg = error_msg.replace(temp_path, "temp.lean")
            return f"COMPILE ERROR:\n{error_msg}"

        if "warning" in result.stderr.lower():
            warnings = "\n".join([
                line for line in result.stderr.split('\n')
                if "warning" in line.lower()
            ])
            return f"COMPILE WARNINGS:\n{warnings}"

        return "pass"

    except subprocess.TimeoutExpired:
        return "ERROR: Compilation timed out"
    except Exception as e:
        return f"SYSTEM ERROR: {str(e)}"
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def main():
    input_path = 'path/to/input.jsonl'
    output_path = 'path/to/output.json'
    header_path = 'path/to/header.lean'

    with open(header_path, 'r', encoding='utf-8') as f:
        header = f.read()

    with open(input_path, 'r', encoding='utf-8') as file:
        data = []
        for line in file:
            item = json.loads(line)
            outputs = item.get('outputs', [])
            item['compile'] = []

            for output in outputs:
                content = header + "\n" + output
                result = validate_lean_code(content)
                item['compile'].append(result)

            data.append(item)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
