import os
VARIANTS={v:dict(features=list(range(12)),start=0,frames=124,coordinate_condition=v) for v in ['baseline','stable_scale']}
VARIANT=os.environ.get('GDI_VARIANT','baseline')
assert VARIANT in VARIANTS
V=VARIANTS[VARIANT]
NF=12;NT=124
