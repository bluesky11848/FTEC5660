#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Chain shape (LCEL):

        ChatPromptTemplate -> ChatDeepSeek(vision) -> parse -> drop `thinking`

    The model is only ever allowed to *read* a receipt. It never performs
    arithmetic and never composes the final answer string; both of those are
    deterministic Python below. That keeps ``answer_queries`` able to emit a
    response containing exactly one number.

    Model-name note: the homework mandates ``deepseek-v4-flash-vision-exp``, and
    DeepSeek still accepts that legacy name (it is served by the current Flash
    model). Should a deployment ever reject it, set ``HW1_MODEL=deepseek-flash``
    in ``.env`` to switch to the current vision model name -- no code change.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_deepseek import ChatDeepSeek
    from pydantic import BaseModel, Field

    class ReceiptExtraction(BaseModel):
        """Verbatim figures copied off one supermarket receipt."""

        amount_paid_after_rounding: str | None = Field(
            default=None,
            description="The FINAL amount actually paid on this receipt, i.e. the value "
            "printed after any ROUNDING line (e.g. the OCTOPUS / PAID / NET line). "
            "Copy it exactly as printed, e.g. '102.30'.",
        )
        subtotal_after_discounts_before_rounding: str | None = Field(
            default=None,
            description="The SUBTOTAL printed on the receipt: the net amount AFTER "
            "discounts have already been subtracted but BEFORE rounding, e.g. "
            "'102.31'. Never put the pre-discount item sum here.",
        )
        discount_total: str | None = Field(
            default=None,
            description="Total of ALL discount/promotion/coupon/saving lines as a "
            "POSITIVE number. A '-5.39' discount line means '5.39'. Use '0' when "
            "the receipt has no discount line at all.",
        )
        rounding_adjustment: str | None = Field(
            default=None,
            description="The signed value on the ROUNDING line, e.g. '-0.01'. "
            "Use '0' when there is no rounding line.",
        )
        printed_amount_without_discounts: str | None = Field(
            default=None,
            description="Only if the receipt literally prints a 'without discount' / "
            "'original total' figure, copy it here. Otherwise null.",
        )
        line_items: list[str] = Field(
            default_factory=list,
            description="Every purchased line's price as a PLAIN DECIMAL STRING, e.g. "
            "['10.00', '36.90', '60.80']. Never objects, never codes or quantities.",
        )
        discount_lines: list[str] = Field(
            default_factory=list,
            description="Every discount/promotion/coupon line's value as a PLAIN DECIMAL "
            "STRING and a POSITIVE number, e.g. ['5.39']. Never objects.",
        )
        receipt_type: str | None = Field(
            default=None,
            description="Short free-text label for the receipt layout, e.g. 'PARKNSHOP'.",
        )

    system_prompt = (
        "You are a meticulous receipt transcription engine for Hong Kong "
        "supermarket receipts. You ONLY transcribe figures that are visibly "
        "printed on the image. You never invent, estimate, or recalculate "
        "anything. If a field is absent from the receipt, return null (or an "
        "empty list).\n"
        "\n"
        "Critical field semantics -- read carefully:\n"
        "1. SUBTOTAL is the net amount AFTER discounts have already been "
        "subtracted, but BEFORE rounding. It is NOT the sum of the original "
        "item prices. These receipts print it with a Chinese label that looks "
        "like '\u5c0f\u8a08' (often preceded by an item count, e.g. '20 \u5c0f\u8a08' or "
        "'4 \u5c0f\u8a08'), or as 'SUBTOTAL' / '\u5408\u8a08'. READ THAT PRINTED FIGURE "
        "DIRECTLY. Never compute it yourself by adding up the item lines.\n"
        "2. Discount / promotion / coupon lines are printed as negative numbers "
        "and are often placed directly UNDER the item they apply to, not in a "
        "separate section -- so scan the entire item list for them. Labels "
        "include '5% OFF (CU)', '5% OFF (CU-SCO)', 'Buy 2 Save', 'Buy 3 Save', "
        "\u5305\u88dd\u8b8a\u5f62 (packaging damage) and similar. Report their magnitude as a "
        "POSITIVE number. The printed SUBTOTAL already has every one of these "
        "subtracted, so you do not need to reconcile them against it.\n"
        "3. The final amount paid is the figure printed AFTER the ROUNDING "
        "line. On these receipts that is the 'OCTOPUS' line; other receipts may "
        "label it PAID, NET, TOTAL, AMOUNT DUE or CARD. Take whichever total "
        "appears last, i.e. after rounding.\n"
        "4. ROUNDING is a small adjustment; it is never a discount. It is "
        "usually a few cents but can reach about 0.10.\n"
        "\n"
        "Worked example you must reproduce exactly (this is receipt5.jpg in the "
        "public set):\n"
        "  item lines 10.00, 36.90, 60.80; then a line '5% OFF (CU-SCO) -5.39'; "
        "then '4 \u5c0f\u8a08 102.31'; then 'ROUNDING -0.01'; then 'OCTOPUS 102.30'.\n"
        "  -> amount_paid_after_rounding = '102.30'\n"
        "  -> subtotal_after_discounts_before_rounding = '102.31'\n"
        "  -> discount_total = '5.39'   (positive)\n"
        "  -> rounding_adjustment = '-0.01'\n"
        "  -> line_items = ['10.00', '36.90', '60.80']\n"
        "  -> discount_lines = ['5.39']\n"
        "\n"
        "Output discipline: return only the structured fields. Do not write "
        "explanations, markdown, percentages, dates, store numbers, member "
        "numbers, or item quantities. Amounts only, as plain decimal strings "
        "with no currency symbol and no thousands separator."
    )

    # A short instance-shaped example, NOT the full JSON Schema. Dumping
    # pydantic's schema put `properties` / `anyOf` / `title` into the prompt, and
    # the model sometimes answered with an echo of that schema instead of an
    # instance -- the cause of intermittent whole-run failures.
    human_prompt = (
        "Transcribe the figures from this receipt.\n"
        "If the image is unreadable or is not a receipt, return null for every "
        "amount field and an empty list for every list field.\n"
        "\n"
        "Reply with a single JSON object and nothing else -- no prose, no "
        "markdown fences, and do not restate any schema. Use these exact 8 "
        "keys:\n"
        "{{\n"
        '  "amount_paid_after_rounding": "102.30",\n'
        '  "subtotal_after_discounts_before_rounding": "102.31",\n'
        '  "discount_total": "5.39",\n'
        '  "rounding_adjustment": "-0.01",\n'
        '  "printed_amount_without_discounts": null,\n'
        '  "line_items": ["10.00", "36.90", "60.80"],\n'
        '  "discount_lines": ["5.39"],\n'
        '  "receipt_type": "FUSION"\n'
        "}}\n"
        "\n"
        "Every amount is a plain decimal string: no currency symbol, no "
        "thousands separator, no percent sign. The values above show the shape "
        "only -- replace them with the figures printed on THIS receipt."
    )

    # Images must live in the human message: DeepSeek rejects images sent in a
    # system message with a 400. The native OpenAI `image_url` block shape is
    # used because it needs no client-side conversion.
    import os

    # JSON mode, not tool calling. This model runs in thinking mode by default,
    # and thinking mode rejects the `tool_choice` that LangChain's
    # ``with_structured_output`` sets -- so that helper returns HTTP 400 for
    # every receipt ("Thinking mode does not support this tool_choice").
    # DeepSeek's JSON Output mode is compatible with thinking (verified against
    # the live API). Keeping thinking ENABLED matters: it is the model's
    # reasoning pass, and turning it off measurably weakens reading dense
    # Chinese item lines.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                [
                    {"type": "text", "text": human_prompt},
                    {"type": "image_url", "image_url": {"url": "{image}"}},
                ],
            ),
        ]
    )

    model_name = os.environ.get("HW1_MODEL", "deepseek-v4-flash-vision-exp")
    # A generous completion budget is required, not optional: this model spends
    # its thinking tokens BEFORE emitting any content, and a dense receipt can
    # burn >11k reasoning tokens. With 8192 the reply is truncated and the
    # request fails with LengthFinishReasonError.
    model = ChatDeepSeek(model=model_name, temperature=0, max_tokens=16384, max_retries=2)
    model = model.bind(response_format={"type": "json_object"})

    def _extract_json(payload: Any) -> dict[str, Any]:
        """Pull the JSON object out of the model reply and unwrap it.

        Two shapes are seen in practice: the instance itself, and -- because the
        prompt quotes the JSON Schema -- a schema echo of the form
        ``{"title": ..., "properties": {...}}``. The echoed field values are
        still the receipt's real figures, so unwrapping rescues that reply
        instead of discarding it.
        """
        content = payload.content if hasattr(payload, "content") else payload
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        if isinstance(content, str):
            start, end = content.find("{"), content.rfind("}")
            if start == -1 or end <= start:
                return {}
            try:
                content = json.loads(content[start:end + 1])
            except (ValueError, TypeError):
                return {}
        if not isinstance(content, dict):
            return {}

        inner = content.get("properties")
        if isinstance(inner, dict) and any(
            key in inner
            for key in (
                "amount_paid_after_rounding",
                "subtotal_after_discounts_before_rounding",
                "discount_total",
            )
        ):
            return inner
        return content

    return prompt | model | RunnableLambda(_extract_json)


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    Stage 1 (chain): vision extraction, one invocation per receipt.
    Stage 2 (this function): deterministic Decimal aggregation.
    Stage 3 (this function): render a response containing exactly one number.
    """
    observations = _observe_receipts(chain, images)
    totals = _aggregate(observations)
    return {query: _render(totals[query]) for query in QUERIES}


# --------------------------------------------------------------------------- #
# Stage 2: deterministic aggregation                                          #
# --------------------------------------------------------------------------- #

_CENT = Decimal("0.01")
_MAX_CONCURRENCY = 4


def _to_decimal(value: Any) -> Decimal | None:
    """Coerce a transcribed amount to a clean 2-decimal Decimal."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        candidate = value
    else:
        text = str(value).strip()
        if not text:
            return None
        text = text.replace(",", "").replace("HK$", "").replace("$", "").strip()
        text = text.replace("(", "-").replace(")", "")
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            candidate = Decimal(match.group(0))
        except InvalidOperation:
            return None
    try:
        return candidate.quantize(_CENT)
    except InvalidOperation:
        return None


def _sum_decimals(values: Any) -> Decimal:
    total = Decimal("0")
    if not isinstance(values, (list, tuple)):
        return total
    for value in values:
        parsed = _to_decimal(value)
        if parsed is not None:
            total += parsed
    return total.quantize(_CENT)


def _normalise(extraction: Any) -> dict[str, Any] | None:
    """Turn one raw model extraction into a reconciled receipt record."""
    if not isinstance(extraction, dict) or not extraction:
        return None

    subtotal = _to_decimal(extraction.get("subtotal_after_discounts_before_rounding"))
    discount = _to_decimal(extraction.get("discount_total"))
    rounding = _to_decimal(extraction.get("rounding_adjustment"))
    printed_without = _to_decimal(extraction.get("printed_amount_without_discounts"))

    item_lines = extraction.get("line_items")
    item_lines = item_lines if item_lines is not None else extraction.get("items")
    items_sum = _sum_decimals(item_lines)
    discount_lines_sum = _sum_decimals(
        extraction.get("discount_lines") or extraction.get("discounts")
    )

    # A discount summary and itemised discount lines are two views of the same
    # money; take the larger so a missed mention never shrinks the discount.
    # Receipts print discounts as negative numbers, so take the magnitude.
    discount = abs(discount) if discount is not None else None
    if discount is None:
        discount = abs(discount_lines_sum) if discount_lines_sum != 0 else None
    elif abs(discount_lines_sum) > discount:
        discount = abs(discount_lines_sum)
    discount = discount if discount is not None else Decimal("0.00")

    # Recount the itemised lines when the model ignored the SUBTOTAL label.
    if subtotal is None and items_sum > 0:
        subtotal = items_sum - discount
    if subtotal is None:
        # Last resort: the pre-rounding net is the paid amount minus rounding.
        paid_guess = _to_decimal(extraction.get("amount_paid_after_rounding"))
        if paid_guess is not None and rounding is not None:
            subtotal = paid_guess - rounding
    if subtotal is None:
        return None

    without_discount = None
    if printed_without is not None:
        expected = subtotal + discount
        if abs(printed_without - expected) <= Decimal("1.00"):
            without_discount = printed_without

    # Q2 is "SUBTOTAL plus every discount added back". Both routes to it are
    # imperfect in different directions, so take the larger one:
    #   * subtotal + discount_total understates Q2 when a discount line was
    #     missed -- measured on the public set, the discount list dropped a
    #     $1.00 line on receipt2 while the item prices were read perfectly.
    #   * items_sum alone risks over-counting if a non-purchase line is picked
    #     up, so it is only trusted when it stays close to the other route.
    direct = subtotal + discount
    if without_discount is None:
        without_discount = direct
    if items_sum > 0 and items_sum > without_discount:
        drift = items_sum - direct
        tolerance = min(Decimal("5.00"), (direct * Decimal("0.05")).quantize(_CENT))
        if abs(drift) <= tolerance:
            without_discount = items_sum
    if without_discount > direct:
        discount = without_discount - subtotal

    # Reconcile the final payment with the pre-rounding net.
    #
    # The gap between "subtotal" and "paid" is the rounding adjustment, which is
    # small by construction: real receipts round to the nearest 0.05 or 0.10, so
    # |gap| stays well under 1.00. A gap larger than that means the model grabbed
    # the wrong total -- usually the pre-discount figure -- and must not be
    # trusted. The printed ROUNDING line, when present, is authoritative.
    paid = _to_decimal(extraction.get("amount_paid_after_rounding"))
    if rounding is not None and abs(rounding) > Decimal("1.00"):
        rounding = None  # too large to be a rounding line; ignore it
    if rounding is not None and rounding == 0 and paid is not None:
        # A ROUNDING line of exactly 0.00 on a reconciliation that needs a
        # non-zero adjustment is evidence of mis-transcription, so recompute.
        delta = paid - subtotal
        if 0 < abs(delta) <= Decimal("0.50"):
            rounding = delta
    if rounding is None and paid is not None:
        delta = paid - subtotal
        if abs(delta) <= Decimal("0.50"):
            rounding = delta  # model omitted the line; recover it from `paid`
    if rounding is None:
        rounding = Decimal("0.00")

    if paid is None or abs(paid - (subtotal + rounding)) > Decimal("0.50"):
        paid = subtotal + rounding

    return {
        "subtotal": subtotal,
        "discount": discount,
        "rounding": rounding if rounding is not None else Decimal("0.00"),
        "paid": paid,
        "without_discount": without_discount,
        "items_sum": items_sum,
        "self_consistent": items_sum == 0 or abs(items_sum - without_discount) <= Decimal("0.05"),
    }


def _aggregate(observations: list[dict[str, Any]]) -> dict[str, Decimal]:
    """Sum every reconciled receipt into one amount per query."""
    paid_total = Decimal("0")
    without_discount_total = Decimal("0")
    for record in observations:
        paid_total += record["paid"]
        without_discount_total += record["without_discount"]
    return {
        QUERY_1: paid_total.quantize(_CENT),
        QUERY_2: without_discount_total.quantize(_CENT),
    }


# --------------------------------------------------------------------------- #
# Stage 3: single-number rendering                                            #
# --------------------------------------------------------------------------- #


def _render(amount: Decimal) -> str:
    """Render an amount so that the response holds exactly one number."""
    for candidate in (f"HK${amount:.2f}", f"HK${amount:,.2f}"):
        if len(_MONEY_RE.findall(candidate)) == 1 and parse_single_amount(candidate) == amount:
            return candidate
    return f"HK${Decimal('0.00'):.2f}"


# --------------------------------------------------------------------------- #
# Stage 1: vision extraction with retries and per-receipt isolation            #
# --------------------------------------------------------------------------- #


def _extract_once(chain: Any, data_url: str) -> Any:
    result = chain.invoke({"image": data_url})
    if hasattr(result, "content"):
        return _parse_jsonish(result.content)
    if isinstance(result, str):
        return _parse_jsonish(result)
    return result


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def _parse_jsonish(text: Any) -> Any:
    if not isinstance(text, str):
        return None
    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except (ValueError, TypeError):
        return None


def _observe_receipts(chain: Any, images: list[Path]) -> list[dict[str, Any]]:
    """Read every receipt, in parallel, with per-image isolation.

    One receipt can never break the run: a failed read is retried on its own and
    finally dropped. Dropping a receipt shifts both totals by that receipt's
    amount, so this is deliberately the last resort.
    """
    prepared: list[tuple[Path, str]] = []
    for path in images:
        try:
            prepared.append((path, image_data_url(path)))
        except OSError:
            continue  # unreadable file: skip, never abort the batch

    raws = _batch_extract(chain, [url for _path, url in prepared])
    if len(raws) != len(prepared):
        raws = [None] * len(prepared)

    records: list[dict[str, Any]] = []
    for (path, data_url), raw in zip(prepared, raws):
        record = _normalise(raw)
        if record is None or not record["self_consistent"]:
            # Spend a second opinion only where the first read did not reconcile.
            for _attempt in range(2):
                try:
                    candidate = _normalise(_extract_once(chain, data_url))
                except Exception:
                    continue
                if candidate is None:
                    continue
                if record is None or (
                    candidate["self_consistent"] and not record["self_consistent"]
                ):
                    record = candidate
                if candidate["self_consistent"]:
                    break
        if record is not None:
            records.append(record)
    return records


def _batch_extract(chain: Any, data_urls: list[str]) -> list[Any]:
    """Best-effort concurrent extraction; returns one slot per input."""
    if not data_urls:
        return []
    payloads = [{"image": url} for url in data_urls]
    try:
        results = chain.batch(
            payloads, config={"max_concurrency": _MAX_CONCURRENCY}, return_exceptions=True
        )
        if len(results) == len(payloads):
            return results
    except Exception:
        pass  # fall through to the sequential path below

    results = []
    for payload in payloads:
        try:
            results.append(chain.invoke(payload))
        except Exception:
            results.append(None)
    return results


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
