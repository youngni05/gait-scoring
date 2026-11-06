import os
VARIANTS={'full124': {'features': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'start': 0, 'frames': 124}, 'common124': {'features': [0, 1, 2, 3, 4, 5, 8, 11], 'start': 0, 'frames': 124}, 'full90': {'features': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'start': 17, 'frames': 90}, 'common90': {'features': [0, 1, 2, 3, 4, 5, 8, 11], 'start': 17, 'frames': 90}}
VARIANT=os.environ.get('GDI_VARIANT','full124')
assert VARIANT in VARIANTS
V=VARIANTS[VARIANT]
NF=len(V['features']);NT=V['frames']
