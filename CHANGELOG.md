# Changelog

## 1.0.0

First release.

- `manyhands run <folder>` transcribes a folder of page images or PDFs with several local
  vision-OCR models, aligns the transcripts, and writes `consensus.txt`, `agreement.json`
  and a self-contained `report.html` per page, plus an index across the folder.
- The report highlights every disagreement in three bands by agreement ratio. Clicking one
  shows the crop of the page beside each model's reading, `n` and `p` step through them, and
  the reports link to each other and to the folder index.
- `manyhands confirm <page>` records accepted readings into `.manyhands/glossary.json`,
  filed by the competing readings so one decision settles every later page that repeats it.
- `manyhands eval <dataset-dir>` scores flags against ALTO or PAGE-XML ground truth and
  reports error-capture rate, flag rate and CER.
- Five permissively-licensed default models, driven through churro-ocr. `stanford-oval/churro-3B`
  is opt-in and prints its licence notice.
- Anchor-based progressive alignment with null tokens, majority voting with agreement ratios,
  and tie-breaking by backend rank then glossary.
- Per-page transcript cache, so an interrupted run resumes and a re-report costs no GPU time.
- Backends that crash, loop, return nothing, or return a wildly different amount of text are
  kept out of the vote and named in the report footer rather than failing the page. One that
  fails to load, or fails three pages in a row, is not tried again for the rest of the run.
