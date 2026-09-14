# PROGRESS

Status: v1.0.0, complete and verified on this machine. Nothing is published. The owner ships
it from the phone.

## Verified working

Every claim here was run, not reasoned about.

- `manyhands run`, `manyhands confirm` and `manyhands eval` are implemented, exercised by
  247 tests, and were each run by hand on real input.
- `pytest -q` → 247 passed. `ruff check .` → clean. difflib clone check at 50%: src 105
  functions, 0 pairs; tests 162 functions, 0 pairs.
- Real ensemble run, no cache, `tests/data/pages` with the default five: 12:16:08 to 12:44:15,
  340 of 427 slots flagged. Logged in `runs/demo-rerun.log`.
- Eval on two HTR-United datasets with the default five. Célestine diary, 4 pages: capture
  84.8%, flag 82.2%, CER 13.2%. tapuscorpus, 4 pages: capture 79.3%, flag 24.2%, CER 63.9%,
  which becomes 48.0% / 21.8% / 24.8% once the title page with 72 characters of ground truth
  is excluded. Both are in the README with the per-page tables and the caveats. These numbers
  moved from the ones 1.0.0 first quoted, because eval now scores the consensus text the run
  writes rather than a differently-line-broken rebuild of it. See the review pass below.
- Clean-venv install from the rebuilt wheel in `D:\tmp\mh-clean3`, Python 3.12, no repo on the
  path, over a page folder outside the repo: `manyhands --version`, `--help`, a real `run`
  with `--json`, a `confirm --slot 0 --reading Samedi`, and a second `run` showing
  `"confirmed": true` on that slot.
- The same clean venv without torch prints the `pip install 'manyhands[hf]'` message and
  exits 3 rather than raising ImportError.
- Reality checks, each run for real: missing folder exits 2, empty folder exits 2 naming the
  extensions it reads, an unreadable image yields `bbox: null` and no crops rather than a
  traceback, a blank page reports `"blank": true` and exits 0, every backend failing exits 3.
- PDF input: a two-page PDF rasterises to `journal-p0001.png` and `journal-p0002.png`, pages
  named `journal.pdf#1` and `#2`, exits 0.
- `release()` frees VRAM between models: 24.0 GB in use during PaddleOCR-VL, 1.2 GB after.
- Cross-platform, run for real rather than assumed: `docker run python:3.12-slim` and
  `python:3.13-slim` over the repo both give `ruff` clean and `244 passed, 3 skipped`. That
  is the matrix `.github/workflows/ci.yml` runs, so CI should be green on the first push.
  Windows 3.13 is the one cell not run here, and the only 3.13-specific behaviour is
  interpreter-level, not platform-level.

## Found by running it on Linux, worth carrying forward

- **churro-ocr's slotted-dataclass `super()` bug is Python 3.12 only.** On 3.13, CPython
  repoints the `__class__` cell when `@dataclass(slots=True)` rebuilds the class, so the same
  churro-ocr 0.3.0 works unpatched. Checked cell by cell in `python:3.12-slim` (False, False)
  and `python:3.13-slim` (True, True). `repair_preset_super()` therefore returns an empty list
  on 3.13, which is what its test now asserts, and `docs/churro-issue.md` says so, since a
  maintainer developing on 3.13 cannot see the bug at all. The LESSONS.md bullet from this
  session is scoped to 3.12.10 so it is not wrong, but it is worth the refinement if you want
  to spend a future bullet on it.
- **CLI tests were width-dependent.** Typer's rich error panel wraps to the terminal, and a
  long Linux `tmp_path` split `does not exist` across two lines with box art in between. The
  `output()` helper in `tests/test_cli.py` now strips the box characters and collapses
  whitespace, so assertions test wording rather than layout.

## Review pass, and what it changed

A full read of the shipped code as a reviewer, each finding proven against the running program
before it was touched. Six things were wrong.

1. **A model named twice voted twice.** `resolve_backends` returned one runner per name, so
   `--backends olmocr,allenai/olmOCR-2-7B-1025` built two voters from one model. Because the
   per-slot readings are keyed by backend name, one stream's token was dropped and the other
   counted twice: a 2-1 split reported as 3-0 unanimous. Proven with three streams reading
   Renwick/Fenwick/Fenwick, which scored `Fenwick 1.0 {'Fenwick': 3}` instead of
   `Fenwick 0.667 {'Fenwick': 2, 'Renwick': 1}`. Now de-duplicated by resolved model id, with
   the reason printed. This is the sharpest of the six: it turned a lone reading into a
   corroborated one, which is the exact claim the tool exists to make.
2. **eval scored text no artifact ever held.** `consensus_spans` started a new line wherever
   the slot's line index changed, while `render_consensus` grouped by line index. A page whose
   slots came back out of line order, which happens because the winning backends break the page
   differently, was scored against a different string from the one in `consensus.txt`. A ballot
   with lines 0,1,0 rendered `alpha gamma\nbeta` and scored `alpha\nbeta\ngamma`.
   Fixed, and the new test asserts the scored text equals `fold(render_consensus(ballot))`
   character for character. Every published eval number moved as a result.
3. **The same line grouping was written four times**, in `vote`, `report`, `pipeline` and
   `evaluate`, and finding 2 is what happens when one copy drifts. Extracted to
   `vote.group_by_line`, which all four now call.
4. **The glossary could never break a tie.** `_break_tie` sorted on backend rank first. A
   backend sits in exactly one reading group, so the best rank per group is unique and always
   decided, leaving the glossary key unreachable. With a confirmed `Fenwick` and a 1-1 tie the
   result was still `Renwick`. Reordered, see the deviations below.
5. **`format_report` took target thresholds nothing passed.** The CLI now has
   `--target-capture` and `--target-flag`, and they reach the `[met]` line and the JSON.
6. **`GroundTruth.image` was parsed and tested and never used.** Pairing is by file stem, so a
   dataset that renamed its images without rewriting the XML silently scored a good transcript
   against the wrong page. `mismatched_image()` now warns on stderr.

Also reshot `docs/report.png`, because the report gutter now numbers display lines to match
`consensus.txt` rather than by raw line index.

One enhancement came out of the same pass: `run --json` writes the folder-level summary that
`agreement.json` cannot, since that file is per page. `report.page_rows` is now the one place a
page is summarised, so the HTML index and the JSON cannot drift the way finding 2 did.

## Known deviations from the brief

- **The brief's target is not met on either dataset.** It asked for ≥70% capture at <20% flag
  rate. Cursive gets the capture and misses the flag rate badly (82.2%). Typescript gets close
  on flag rate (21.8%) and misses the capture (48.0%). The README prints both verbatim and
  says so. This is a result, not an outstanding task: it is what five heterogeneous VLMs
  actually do on this material.
- **Ties break by glossary, then by backend rank.** The brief said rank then glossary. Rank
  cannot tie, so that ordering makes the glossary branch dead code. Proven, then reversed: a
  reading a human already accepted somewhere in this corpus is better evidence than the order
  the models happen to be listed in. `vote.py` says so at the function.
- **The brief said uv/hatchling.** hatchling is the build backend. The README's Development
  section uses `python -m venv` and pip, because that is what the project was built and tested
  with here and an untested command in a README is worse than an unfashionable one.
- **`Development Status :: 4 - Beta` alongside `version = "1.0.0"`.** Deliberate. 1.0.0 means
  the CLI and the file formats are what I intend to support. Beta means eight scored pages from
  two documents is not enough evidence to call the results stable. The README Status section
  says this out loud.
- **dots.ocr-1.5 stays in the default five** even though it contributed zero votes to all ten
  pages run here: no text on the samples, CUDA OOM on all eight eval pages. Removing a model
  the brief named, on evidence from two documents and one 24 GB card, would be overfitting to
  this sample. Instead it is documented, and a backend that fails three pages in a row is now
  dropped for the rest of the run so it cannot charge for every page.
- **The README image and the churro issue link are absolute GitHub URLs** pointing at
  `Booyaka101/manyhands` on `main`. PyPI does not rewrite relative links, so they would render
  broken on the project page. They resolve once the repo is pushed under that name.

## Not built for v1, and why

- **Cross-column reading order.** v1 aligns per detected line block and documents the limit.
  Doing it properly needs a layout model, which is a second class of dependency.
- **Tight per-word boxes.** churro-ocr's interface is text only, so there are no coordinates
  to use. The report shows the line crop with the slot marked along it.
- **Confidence from model logits.** Would mean bypassing churro-ocr's own generate call, which
  is the one thing the brief said not to reimplement. Agreement between models is the signal
  this tool is built on anyway.
- **A `--resample` path.** Ruled out on measurement, not taste: four of the five default churro
  profiles pin `do_sample: False`, so sampling one model five times returns one answer.
- **Parallel backends on one GPU.** The default five do not fit in 24 GB together, which is why
  the run loop is backend-major.
- **A diff view against an existing transcription.** The obvious next feature. An archive with
  a partial transcript wants the models scored against it, not just against each other.
- **Resume within a backend after Ctrl-C.** The transcript cache already makes a re-run cheap,
  but the progress line restarts from page one.

## If you pick this up

- `runs/` is scratch, gitignored, and holds every log and eval report quoted in the README.
- `tests/data/pages/.manyhands/transcripts/` has both the three committed `stub-*` fixtures and
  my own full-ensemble caches. `.gitignore` commits only the stubs. `SOURCE.md` in that folder
  says exactly which two fields were edited to make the stubs.
- `docs/churro-issue.md` is a drafted upstream bug report about the `@dataclass(slots=True)`
  plus zero-argument `super()` crash in four churro-ocr 0.3.0 presets. It is drafted, not filed.
  Nothing goes upstream without the owner's go.

## Single best first distribution step

PyPI, then one post. Publish the wheel and sdist in `dist/` with `twine upload`, because every
other route needs an install command that works. Then one write-up in r/LocalLLaMA that leads
with the measured numbers including the bad ones: 48.0% capture on typescript, an 82% flag rate
on cursive, and dots.ocr contributing nothing on a 24 GB card. That audience has the GPUs to
reproduce it and no patience for a launch post that quotes only the 79.3%. Read 10-20 recent
posts there first and match the register, and nothing gets posted without the owner's go.
