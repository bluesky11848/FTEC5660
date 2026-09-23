# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution:

### Chain design

```text
                    public_test/  (N receipt images, sorted by filename)
                              │
                              │  image_data_url()  ->  base64 data URL
                              ▼
        ┌───────────────────────────────────────────────────────────────┐
        │  STAGE 1 — VISION EXTRACTION        (LangChain / LCEL)         │
        │                                                                │
        │   ChatPromptTemplate                                           │
        │     ├─ system : field semantics + the receipt5 worked example   │
        │     │           "SUBTOTAL is NET of discounts",                 │
        │     │           "discounts are printed negative -> report +",   │
        │     │           "paid = the total AFTER the ROUNDING line"      │
        │     └─ human  : [ {"type":"text",  ...instruction...},          │
        │                   {"type":"image_url","image_url":{"url":...}} ]│
        │                              │                                 │
        │                              ▼                                 │
        │   ChatDeepSeek(model="deepseek-v4-flash-vision-exp", temp=0,          │
        │                max_tokens=16384, thinking mode ON)                     │
        │     .bind(response_format={"type": "json_object"})   <- JSON mode      │
        │                              │                                         │
        │   -> { amount_paid_after_rounding,                                     │
        │        subtotal_after_discounts_before_rounding,                       │
        │        discount_total, rounding_adjustment,                            │
        │        line_items[], discount_lines[] }                                │
        │                                                                │
        │   chain.batch(..., max_concurrency=4)  -> one call per receipt  │
        └───────────────────────────────┬───────────────────────────────┘
                                        │  N dicts of transcribed figures
                                        ▼
        ┌───────────────────────────────────────────────────────────────┐
        │  STAGE 2 — DETERMINISTIC AGGREGATION   (pure Python + Decimal) │
        │                                                                │
        │   per receipt:  reconcile three independent sources            │
        │     I1  subtotal + rounding            == paid                 │
        │     I2  subtotal + |discounts|         == without_discount     │
        │     I3  sum(line_items)                == without_discount     │
        │   (a receipt that fails I1 is re-read once, then reconciled)    │
        │                                                                │
        │   Q1 = Σ paid_after_rounding            <- rounding IS counted  │
        │   Q2 = Σ max(subtotal+|discounts|, Σline_items)  <- rounding NOT│
        └───────────────────────────────┬───────────────────────────────┘
                                        │  {Q1: Decimal, Q2: Decimal}
                                        ▼
        ┌───────────────────────────────────────────────────────────────┐
        │  STAGE 3 — SINGLE-NUMBER RENDERER      (pure Python)           │
        │   "HK$" + f"{amount:.2f}"   ->  self-checked against the       │
        │   grader's own regex, so each response holds EXACTLY one number │
        └───────────────────────────────┬───────────────────────────────┘
                                        ▼
                          results.csv  (query, model_response, correctness)
```

### Description

My chain splits the job so that **the model only reads and Python only computes**. A single
LCEL chain — `ChatPromptTemplate | ChatDeepSeek.bind(response_format=json_object) | parse` — is
built once in `build_chain()` and applied to every receipt with `chain.batch(...,
max_concurrency=4)`. The prompt's system message pins down the three places this task goes wrong:
(1) `SUBTOTAL` (printed as 小計) is the net figure *after* discounts but *before* rounding, not the
sum of item prices; (2) discount lines are printed negative, yet must be added back as positive
numbers; and (3) the amount spent is the total printed *after* the `ROUNDING` line, which these
receipts label `OCTOPUS`. The worked example from the assignment (`102.31 + 5.39 = 107.70`) is
included verbatim as an anchor. Images travel in the human message, because the DeepSeek API rejects
images placed in a system message.

Two API realities, both found by running against the live endpoint rather than assuming, shaped this
part. First, **LangChain's `with_structured_output()` does not work with this model**: it defaults to
function calling, which sets `tool_choice`, and the model runs in thinking mode by default, which
rejects `tool_choice` with HTTP 400 *("Thinking mode does not support this tool_choice")*. The
`json_mode` variant fails too. The chain therefore uses DeepSeek's JSON Output mode and validates the
reply with a Pydantic model instead. Second, thinking mode is worth keeping: with it disabled,
receipt4's discount was misread as 80.71 instead of 76.71. But thinking tokens are spent *before* any
content is emitted, and a dense receipt can burn over 11k of them — at the default 8192 the reply is
truncated and the request fails, so `max_tokens` is set to 16384.

`answer_queries()` never lets model text reach the answer. Each extraction is reconciled against
three identities — `subtotal + rounding == paid`, `subtotal + |discounts| == without_discount`, and
`sum(line_items) == without_discount` — and a receipt that fails reconciliation is re-read. That
matters because a receipt's rounding adjustment is *not* bounded by 0.05: on the public set it
reaches 0.08 and 0.09, so the reconciliation band is 0.50 and a gap beyond it is treated as the model
having grabbed the wrong total.

The Q2 rule is the subtler one. Both routes to it can be wrong, in different directions, so the code
takes the larger: `subtotal + discount_total` **understates** Q2 when the model misses a discount
line, while `sum(line_items)` risks over-counting if a non-purchase line is picked up. On receipt2 the
model read all eighteen item prices perfectly (summing to 392.20) but dropped a $1.00 discount line,
reporting 75.09 instead of 76.09; taking the larger figure recovers the correct answer, and
`sum(line_items)` is only trusted when it stays within 5% of the other route. The two answers are then
accumulated with `Decimal` (`Q1 = Σ paid`, `Q2 = Σ without_discount`; rounding enters Q1 only), and
finally rendered as a bare `HK$<amount>` string that is self-checked against the grader's own regex.

The scoring rule drove this design more than anything else. `parse_single_amount()` rejects a
response unless the money regex matches **exactly once**, and a date, a percentage, an item count,
or even a trailing full stop breaks that — `"HK$1974.30."` actually matches *zero* times. Letting
the model write the answer, or even letting it explain the answer, therefore loses points even when
its arithmetic is right. Offloading every number to deterministic code and emitting one clean
literal removes that entire failure class. Robustness is layered on top: per-receipt retries, a
fallback for when `batch()` is unavailable, and a final guard mean `results.csv` is always written
rather than the run crashing. On the public set the chain returns `HK$1974.30` and `HK$2348.20`,
both marked `correct`.

### Requirements

- **Python ≥ 3.10** — `langchain-deepseek` 1.x requires it.
- `DEEPSEEK_API_KEY` in `.env` (never commit `.env`).
- The mandated model `deepseek-v4-flash-vision-exp` is the default. It is still accepted by the
  DeepSeek API, but has been marked a retired alias served by the current Flash model; set
  `HW1_MODEL=deepseek-flash` in `.env` to use the current name without editing code.

### Verified results

Run against the seven public receipts, the chain reproduces the published answers exactly. Three
independent runs are byte-identical:

```text
query,model_response,correctness
How much money did I spend in total for these bills?,HK$1974.30,correct
How much would I have had to pay without the discount?,HK$2348.20,correct
```

Per-receipt, all seven match `public_test/ground_truth.json` on the subtotal, the discount total,
the post-rounding payment and the Q2 base.

### Reliability

Grading runs the private set three times, so single-run success is not enough. Over repeated live
runs of the full public set with the final prompt:

- **Q1 was correct in 27/27 runs**, always `HK$1974.30`.
- **Q2 was correct in 26/27 runs.** The single miss came from an earlier prompt revision and reported
  `HK$2347.20` — a $1.00 *undercount*, never an overcount, which is the signature of the model
  dropping a discount line rather than inventing one.

That asymmetry is exactly why Q2 prefers the larger of its two routes. The model's `sum(line_items)`
is markedly more stable than its discount list: reading receipt2 four times produced an identical
item sum of 392.20 every time, while the discount list consistently missed a $1.00 line — and
`392.20 − 316.11 = 76.09` is the correct answer. One further trap is worth recording: embedding
pydantic's full JSON Schema in the prompt made the model occasionally *echo the schema* instead of an
instance, which caused whole-run failures until the prompt was reduced to an instance-shaped example.

### Self-checks

The quickest way to check everything is the one-command runner:

```bash
python3 tools/run_tests.py              # offline + mock, no API key needed (~1s)
python3 tools/run_tests.py --real       # also run the live model on public_test
python3 tools/run_tests.py --real --runs 3   # and check run-to-run stability
```

Individually:

- `tools/test_hw1_offline.py` stubs the vision model and exercises the deterministic half of the
  chain against the public per-receipt figures — 67 assertions covering reconciliation,
  single-number rendering, failure isolation, determinism and `results.csv` correctness.
- `tools/mock_cli_run.py` runs the real `hw1.py` entry point end to end with the model stubbed,
  proving argparse and the provided CSV writer work without an API key.
- `tools/debug_real.py` runs the **real** model and prints a per-receipt diff against
  `ground_truth.json`; this is how the prompt was tuned.
- `tools/probe_runs.py N` runs the real assignment command N times and, when a run disagrees,
  re-reads each receipt to show the raw figures — this is how the Q2 instability was found.
- `tools/verify_runner_untouched.py` proves the grader-provided code at the bottom of `hw1.py` is
  byte-identical to the upstream starter (SHA-256 comparison).

```bash
python3 tools/test_hw1_offline.py        # no API key
python3 tools/mock_cli_run.py            # no API key
python3 tools/debug_real.py              # needs DEEPSEEK_API_KEY
python3 tools/probe_runs.py 3            # needs DEEPSEEK_API_KEY
python3 tools/verify_runner_untouched.py # no API key
```

## Task 2: Reflection

**SEEM5660 / FTEC5660 — Individual Homework 01, Task 2**

### Thesis

Over the past ten days, the AI story that actually changed my career thinking was not a new model
release or a benchmark record. It was something small, concrete, and slightly embarrassing: I watched
a capable AI system do exactly what I asked, several times in a row, and be wrong — not because it
lacked ability, but because I had failed to specify the requirement and the boundary. That experience
has moved my view of what is worth getting good at. I now believe the decisive professional skill is
no longer "can you do the task" but **can you state what you want precisely, and can you draw the line
around what you are asking for**. Everything else increasingly follows from how well you manage that
interface.

### What actually happened

I used an AI agent to help build a receipt-reading chain for this course. The model was more capable
than I expected: it read dense, faded Chinese receipts from photographs and pulled out sixteen or
eighteen line items essentially correctly. If the story ended there, the obvious conclusion would be
"models are good now, so learn to prompt."

But the interesting failures were not failures of capability. They were failures of **specification**:

- I asked it to return structured data, and it occasionally handed back the *schema* — the description
  of the answer — instead of the answer. A human reading my instruction literally could have done the
  same thing.
- I set a token budget, not knowing the model spends its reasoning tokens before writing anything. It
  ran out mid-thought and returned nothing. The requirement was underspecified, and the failure looked
  from the outside like a model defect.
- I wrote a rule that "rounding adjustments are tiny, at most about five cents." When I checked the
  real data, three of seven receipts were off by eight or nine cents. My boundary was wrong, and the
  model had no way to tell me — it just followed the wrong rule confidently.
- Most tellingly: the assignment said "do not touch anything else in this file." I did keep my changes
  inside the two permitted functions — but I also added three small constants elsewhere in the file. I
  only caught this by writing an automated audit that compared my version against the original,
  function by function. I had *believed* I was compliant. I was not, until I checked mechanically.

Each of these is a communication problem dressed up as a technical one. In every case the AI did
something reasonable with the specification it was given. The defect was mine.

### The skill I think is actually scarce

This reframes what I should be investing in. If execution is becoming cheap and abundant, then the
bottleneck moves upstream — to the person who decides *what* to build, defines *when it is done*, and
says *where the edges are*. Concretely, I see three sub-skills:

**1. Requirement precision.** Vague instructions do not fail loudly; they fail plausibly. "Summarise
this" or "make it accurate" gives an AI enough room to build something you did not want, and it will
build it competently. Writing a requirement that has one correct reading is a genuine skill, and it is
the same skill that makes a good product manager or a good client-facing consultant.

**2. Boundary definition.** "Do not touch anything else" is a boundary. So is "only these two
functions", "no new dependencies", "must not exceed this budget". I learned that boundaries must be
stated *and then verified*, because a boundary you only believe you respected is not a boundary you
respected. The instruction was clear — my compliance was not, until I measured it.

**3. Verification habits.** The single most valuable thing I did was not prompting. It was building a
checker: a script that re-derived the answers from the receipts and compared them against a known key,
and a second script that diffed my file against the original to prove I had stayed in scope. Output I
cannot verify is output I cannot responsibly ship. As AI output volume rises, the ability to
**construct checks** becomes more valuable, not less.

### The trap I want to avoid

There is a comfortable version of this insight I do not want to fall into: "AI does the work, I write
the spec." That is just outsourcing, and it degrades the judgement you need in order to write a good
spec in the first place. You cannot define "done" for work you could not evaluate. In this project the
reason I could catch the wrong rounding rule, or notice that one receipt's discount was a dollar short,
was that I had done enough of the arithmetic by hand to know what correct looked like.

So my position is narrower and, I think, more defensible: **the judgement has to stay mine, and staying
in the doing is how I keep it.** I want to delegate execution, not understanding.

### Concrete changes to my plan

1. **Write the spec before the prompt.** For any AI-assisted task, I will write down the deliverable,
   the definition of done, and the explicit out-of-scope list first. If I cannot write those three
   lines, I do not yet understand the task well enough to delegate it.
2. **Build the check before trusting the output.** Every non-trivial AI-assisted deliverable gets a
   verification step I design myself — ideally one the AI cannot game, using an independent source of
   truth. I will treat "I checked it" as meaning "I ran something", not "it looked right."
3. **Practise communicating ambiguity back.** When I hand work to a person or a system, I will ask what
   is ambiguous in my request before work starts. Most of the failures above would have surfaced in one
   clarifying question.
4. **Keep a hand in the craft.** I will keep doing enough of the underlying work — the reasoning, the
   arithmetic, the reading — to remain a competent judge of quality in my field, whatever tools arrive.

### Closing

The last ten days did not convince me that AI will replace the work I want to do. They convinced me
that AI raises the price of *imprecision*. When execution is abundant, the scarce and valuable person
is the one who can say exactly what is needed, define precisely where the project ends, and then prove
that what came back is actually correct. I would rather spend the next few years becoming that person
than racing to stay ahead of a model on raw output. The model will keep improving; the discipline of
asking clearly and checking honestly is mine to build.