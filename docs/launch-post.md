# r/LocalLLaMA drafts

Drafted, not posted. Nothing goes to a community without the owner's go, and the owner owns
the final wording.

Venue research behind these, done 2026-09-14 against the live sub:

- 823,672 subscribers. Flair is required. The project flair is `I Built A Thing`.
- Rule 3 bans "completely/primarily LLM generated copy". Rule 4 is the 1/10th self-promotion
  guideline and requires affiliation to be disclosed.
- Register in the top project posts of the month: link near the top, first person, contractions,
  exact hardware, exact numbers, what failed stated plainly, no headers in the short ones, no
  tidy closing line.
- There is a 20-day-old sticky, "Best Local Vision Language Models - August 2026", 89 comments,
  asking for measured VLM results by VRAM tier. Nobody in it has posted document-OCR numbers.

## Draft A, comment on the vision sticky

Prefer this one. It is a real contribution to a thread that asked the exact question this
project has data on, it does not depend on the account's thin posting history, and it is
much harder to read as self-promotion.

---

Narrow slice of this, document OCR on handwriting, single 4090 so M tier. I ran five
open-weight OCR VLMs over the same pages and scored them against HTR-United ground truth, so
these are measured rather than vibes.

Would keep: olmOCR-2-7B and PaddleOCR-VL-1.5. Nanonets-OCR2-3B is close behind and much
cheaper to load. MinerU2.5-1.2B is fast and mostly fine but it loops, on one page it returned
1725 tokens where every other model returned about 600, same text over and over.

Would not: dots.ocr-1.5 gave me nothing at all. No text on my sample pages, CUDA OOM on all
eight eval pages at 24GB. Someone above rates dots.mocr, the 1.8B, and I believe them, but the
1.5 is a different animal and it did not fit.

What surprised me is how far apart they are on the same line. Consensus of all five on 1914
French cursive still sits at 13.2% CER, and where they disagree they disagree wildly, one word
came back as si, sion, mon and son from four different models. Typescript was better but not by
as much as I expected, 12.9% to 33.3% CER across four pages of a rough scan.

I built a thing around that, mine, MIT, runs several of them and highlights the words they
didn't agree on: github.com/Booyaka101/manyhands. The numbers above stand on their own though
if you just want to know which to load.

---

## Draft B, standalone post, flair `I Built A Thing`

Title:

    I ran five open-weight OCR models over the same 1914 diary and had them vote. 85% of the
    wrong characters land in a word the vote flagged.

Body:

---

github.com/Booyaka101/manyhands, MIT, mine.

A single OCR model transcribing a handwritten page gives you a wall of confident text and no
idea which bits are wrong. So I ran the page through five different local models and
highlighted every word they didn't all agree on. You proofread the highlights instead of the
page.

Not a new idea. Calamari-OCR has had multi-model confidence voting for years, but that wants
trained CTC models with per-glyph probabilities. This just takes five off-the-shelf vision OCR
models and votes on their plain text.

Default five, all open weights, driven through churro-ocr: olmOCR-2-7B, PaddleOCR-VL-1.5,
MinerU2.5-1.2B, Nanonets-OCR2-3B, dots.ocr-1.5. One 4090. They don't fit together, so it loads
one, does the whole folder, unloads, next.

The numbers, eight pages from two HTR-United datasets:

1914 French cursive, four pages. 84.8% of the wrong characters landed inside a flagged word.
But it flagged 82.2% of the words to get there, against a 13.2% consensus CER. At that error
rate almost every word is worth looking at, so that flag rate is honest rather than useful.

Typescript, four pages. Headline is 79.3% capture at a 24.2% flag rate, which looks much
better, except 61% of that dataset's errors sit on one title page with 72 characters of ground
truth. Drop it and it's 48.0% capture at a 21.8% flag rate over a 24.8% CER. My own target was
70% capture under a 20% flag rate. Neither dataset hit it.

dots.ocr-1.5 contributed literally nothing on my card, no text on the samples and CUDA OOM on
all eight eval pages. I left it in the defaults anyway, because pulling a model on evidence
from two documents and one GPU is overfitting to my sample, but it's documented, and any
backend that fails three pages running gets dropped for the rest of that run.

The report is a self-contained html file. Click a highlighted word and you get the crop of the
line next to what each model read, n and p step through them. There's a confirm step that
records your decision in a glossary, so the same disagreement settles once for a whole corpus.

Caveats, because this sub will find them anyway. Eight scored pages, two documents, one hand
each, one GPU. Nothing here has met a hand it wasn't developed against. Needs CUDA and about
16GB of VRAM, and the first run pulls about 34GB of weights.

pip install manyhands

---
