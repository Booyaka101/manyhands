# Sample pages

Two pages from the wartime journal of Célestine Doniau-Danest, digitised by the
Archives départementales du Val-de-Marne and published through HTR-United.

- Source: https://github.com/HTR-United/dataset-celestine-doniau-danest
- Licence: CC-BY-4.0
- Files: `journal-p001.jpg` (title page), `journal-p002.jpg` (25 to 29 July 1914)

## Stub transcripts

`.manyhands/transcripts/stub-a`, `stub-b` and `stub-c` are real readings of these two
pages, captured on an RTX 4090 and frozen so the test suite runs without a GPU. They are
test fixtures, not shipped data: `manyhands run` never reads them unless you point it at
this folder with `--backends stub-a,stub-b,stub-c`.

| stub | model |
| --- | --- |
| stub-a | allenai/olmOCR-2-7B-1025 |
| stub-b | PaddlePaddle/PaddleOCR-VL-1.5 |
| stub-c | opendatalab/MinerU2.5-2509-1.2B |

The text, timings and notes are as the models produced them, which is why stub-b carries a
page-timeout note on `journal-p002`: PaddleOCR-VL hit the 600s generation deadline on the dense
page and still took 810s end to end. Two fields were edited: `backend`, so the stub has a name
that is not a model, and the one-off weight-load note, dropped because a transcript replayed
from the cache never pays for a load.
