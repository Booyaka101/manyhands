# Upstream issue draft: zero-argument super() in slotted preset backends

Not filed yet. This is the draft to post at https://github.com/stanford-oval/churro/issues
before or alongside the manyhands announcement, because two of the five models manyhands
defaults to are unusable without the workaround in `src/manyhands/backends.py`.

Workaround shipped here: `repair_preset_super()`, called from `OCRRunner._load()`.
Delete it once a churro-ocr release contains the fix, and drop the `<1` pin down to that
version.

---

**Title:** `dataclass(slots=True)` breaks `super()` in four preset backends (0.3.0)

Four of the preset backends in `churro_ocr/providers/hf.py` raise on the first page:

```
TypeError: super(type, obj): obj must be an instance or subtype of type
  File ".../churro_ocr/providers/hf.py", line 1360, in _get_processor
```

It hits `PaddleOCRVL15OCRBackend` and `DotsOCR15OCRBackend`, which are the two I ran into,
and by inspection the same pattern is in `ChandraOCR2OCRBackend._get_processor` and
`LFM25VLOCRBackend._get_processor` / `_get_model`. Nothing GPU specific, it fails before
any weights load.

Cause is the interaction between `@dataclass(slots=True)` and zero-argument `super()`.
A slotted dataclass cannot be modified in place, so `dataclasses` builds a *new* class
object and returns that. The `__class__` cell the class body closed over still points at
the original class that got thrown away, and `super()` with no arguments reads that cell,
so the instance is not an instance of the class `super` was handed.

It only shows up on 3.12. On 3.13 the same churro-ocr 0.3.0 works unpatched, because CPython
now repoints the `__class__` cell when the slotted class is rebuilt. I checked the cell directly
in both, same wheel, `python:3.12-slim` and `python:3.13-slim`:

```
py 3.12.14  _get_processor cell is class: False   _get_model cell is class: False
py 3.13.15  _get_processor cell is class: True    _get_model cell is class: True
```

So if you develop on 3.13 this is invisible, and `requires-python = ">=3.12"` means users will
hit it.

Minimal repro on 3.12.10:

```python
from dataclasses import dataclass

@dataclass(slots=True)
class Base:
    def hello(self): return "base"

@dataclass(slots=True)
class Child(Base):
    def hello(self): return super().hello()

Child().hello()   # TypeError
```

Fix is to spell the arguments out, `super(PaddleOCRVL15OCRBackend, self)._get_processor()`,
in each of the affected overrides. Dropping `slots=True` on those classes works too but
costs the memory saving. Happy to send a PR against whichever you prefer.

For anyone hitting this before a release, the cell can be repointed at import time. All
methods in one class body share a single `__class__` cell, so one write per class fixes
every override in it:

```python
from churro_ocr.providers import hf

for cls in (hf.PaddleOCRVL15OCRBackend, hf.DotsOCR15OCRBackend):
    for attribute in vars(cls).values():
        code = getattr(attribute, "__code__", None)
        if code and "__class__" in code.co_freevars:
            attribute.__closure__[code.co_freevars.index("__class__")].cell_contents = cls
            break
```

churro-ocr 0.3.0, Python 3.12.10 on Windows 11, and 3.12.14 / 3.13.15 in Docker.
