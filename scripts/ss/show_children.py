import os

root = r"C:\Users\usuario\Documents\TrebleSDK"

for current, dirs, files in os.walk(root):
    # Skip the venv_treble folder completely
    if "venv_treble" in current:
        continue

    level = current.replace(root, "").count(os.sep)
    indent = "    " * level
    print(f"{indent}{os.path.basename(current)}/")
    subindent = "    " * (level + 1)
    for f in files:
        print(f"{subindent}{f}")
