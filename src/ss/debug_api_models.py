import sys
try:
    import treble_tsdk.client.api_models as ap
    print('== API MODELS FOUND ==')
    print([a for a in dir(ap) if a[0].isupper()])
except ImportError as e:
    print(f'API Models module failed to import: {e}')
    sys.exit(1)
except Exception as e:
    print(f'An unexpected error occurred during inspection: {e}')
    sys.exit(1)