# Save this as 'inspect_tsdk.py' and run it from your venv
import sys
from pathlib import Path

# --- Treble SDK import ---
try:
    from treble_tsdk import treble
    print("Using treble_tsdk\n")
except ImportError:
    print("treble_tsdk is not installed/active in this venv.")
    sys.exit(1)

# Initialize the TSDK client
# NOTE: This will require valid credentials (tsdk.cred) to succeed.
try:
    tsdk = treble.TSDK()
except Exception as e:
    print(f"Could not initialize TSDK client. Check credentials/connection: {e}")
    sys.exit(1)

# List all attributes and methods on the TSDK object
all_attributes = sorted(dir(tsdk))

# Filter out the standard Python dunder methods/attributes for a cleaner list
sdk_attributes = [
    attr for attr in all_attributes 
    if not attr.startswith('__')
]

print("--- TSDK Client (treble.TSDK()) Attributes/Methods ---")
for attr in sdk_attributes:
    # Use getattr() to fetch the object and check if it's a library (i.e., callable or not)
    try:
        obj = getattr(tsdk, attr)
        # Check if it's a callable function (a method) or an object (a library)
        attr_type = "Method" if callable(obj) else "Library/Object"
    except AttributeError:
        # Should not happen, but safe check
        attr_type = "Unknown"
    
    print(f"{attr:<30} | {attr_type}")
    
print("-" * 45)
print("Look for *Library and *get_ methods.")