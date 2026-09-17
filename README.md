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
        │   ChatDeepSeek(model="deepseek-v4-flash-vision-exp", temp=0)    │
        │     .with_structured_output(ReceiptExtraction)   <- JSON schema │
        │                              │                                 │
        │   -> { amount_paid_after_rounding,                             │
        │        subtotal_after_discounts_before_rounding,               │
        │        discount_total, rounding_adjustment,                    │
        │        line_items[], discount_lines[] }                        │
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
        │   Q2 = Σ (subtotal + |discounts|)       <- rounding NOT counted │
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
LCEL chain — `ChatPromptTemplate | ChatDeepSeek.with_structured_output(ReceiptExtraction)` —
is built once in `build_chain()` and applied to every receipt with `chain.batch(...,
max_concurrency=4)`. The prompt's system message pins down the three places this task goes wrong:
(1) `SUBTOTAL` is the net figure *after* discounts but *before* rounding, not the sum of item
prices; (2) discount lines are printed negative, yet must be added back as positive numbers; and
(3) the amount spent is the total printed *after* the `ROUNDING` line, which receipts label
`OCTOPUS`/`PAID`/`NET`. The worked example from the assignment (`102.31 + 5.39 = 107.70`) is
included verbatim as an anchor. Structured output forces the reply into a fixed field set, so the
model can never emit prose. Images travel in the human message, because the DeepSeek API rejects
images placed in a system message.

`answer_queries()` never lets model text reach the answer. Each extraction is reconciled against
three identities — `subtotal + rounding == paid`, `subtotal + |discounts| == without_discount`, and
`sum(line_items) == without_discount` — and a receipt that fails reconciliation is re-read once.
That matters because a receipt's rounding adjustment is *not* bounded by 0.05: on the public set it
reaches 0.08 and 0.09, so the reconciliation band is 0.50 and a gap beyond it is treated as the
model having grabbed the wrong total. The two answers are then accumulated with `Decimal`
(`Q1 = Σ paid`, `Q2 = Σ (subtotal + |discounts|)`; rounding enters Q1 only), and finally rendered as
a bare `HK$<amount>` string that is self-checked against the grader's own regex.

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

### Offline self-check

`tools/test_hw1_offline.py` stubs out the vision model and exercises the deterministic half of the
chain against the public per-receipt figures (67 assertions: reconciliation, single-number
rendering, failure isolation, determinism, `results.csv` correctness). It needs no API key:

```bash
python3 tools/test_hw1_offline.py
```

## Task 2: Reflection

### 过去 10 天的 AI 事件如何改变了我的看法与职业规划

**一、从"模型能不能做"转向"系统能不能交付"**

这十天里最让我改变判断的，不是某个新模型的跑分，而是我自己做这份作业时的经历。这个任务的要求
听起来很平常：读收据，报两个总额。但真正决定成败的，不是视觉模型读得准不准，而是一条评分规则
——响应里必须**有且仅有一个数字**。也就是说，模型即使算对了，只要顺口补一句"（7 张收据）"，
或者结尾多一个句号（`HK$1974.30.` 会让正则**零匹配**），分数就是零。

这让我意识到，评价一个 AI 系统的好坏，指标应该是"端到端交付的可靠性"，而不是"模型能力的上限"。
过去我会先问"这个模型能不能看懂收据"；现在我更先问"这条链路在哪一步会不可靠地失败、失败了
会不会整体崩掉"。为此我把架构改成**模型只负责看、Python 只负责算、代码负责输出**，并给每一张
收据加了独立的失败隔离——一张读错只损失一张，不会让整份 `results.csv` 消失。这种"防御性设计"
的思维方式，我觉得比会用某个具体模型更有长期价值。

**二、确定性计算的回归，以及"AI 工程师"的真实含义**

多模态模型已经能稳定把图像变成结构化字段，这部分确实比我预期强。但同样清楚的是：**算术不该交给
模型**。8 张票、几十行的加减，模型心算极易出错，而 `Decimal` 永远不会。一个隐蔽的坑是——
Q2 必须用"舍入前的 SUBTOTAL"作基数，用"舍入后的实付额"会系统性偏低 0.34 元。这种误差模型不会
告诉你，只有把口径写死、用恒等式逐票对账才能发现。

所以我对自己职业规划的第一个修正是：我不再把"AI 工程师"理解为"会调模型的人"，而是**能判断
哪一步该用模型、哪一步必须用确定性代码，并把两者接成可靠系统的人**。会写 prompt 的门槛在快速
下降，而"知道哪里不能相信模型"这件事，反而越来越值钱。

**三、成本与"够用就好"的工程判断**

另外一个现实感受是成本。视觉输入按尺寸折算成 token，一张收据接近 1024 token；多轮重试、双
pass 校验都会成倍放大开销。这让我从"追求最准"转向"在给定预算下最稳"——只在对账失败的收据上
才花第二次调用，而不是无脑全部跑两遍。我原本以为做 AI 应用的主要约束是模型能力，现在看更多
时候是**成本与延迟的工程权衡**，这更接近传统系统设计的思路。

**四、对个人规划的调整**

具体到行动上，我打算做三件事：一是补足**软件工程与评测**的基本功（写单测、构造回归用例、
把口径固化成恒等式），而不只是追新模型；二是坚持"可复现"——`temperature=0`、确定性聚合，
让同一个输入三次评测结果逐字节一致，这在作业里是加分项，在工作中是底线；三是对模型输出保持
**结构化的怀疑**：用 schema 约束、用交叉校验发现不一致，而不是相信一段看起来很流畅的回答。

**五、一句总结**

这十天让我最确定的一点是：AI 让"做出一个能跑的演示"变得非常容易，但也让"做出一个在陌生输入上
不崩、可解释、可复现的系统"显得更稀缺。前者是能力，后者是工程。我希望自己站在后者这一侧，
而这份作业正是一次很小的练习——真正的收获不是那两行金额，而是那条"不信任模型"的检查表。

> _说明：以上为草稿，请按你真实的专业背景、关注方向与想法改写第一人称细节（尤其是第四部分
> 的三条行动计划）。_
