"""LLM-as-a-judge for IF-Banana_Task_v1_text generations.

Directory expectations (per task under `Main/IF-Banana_Task_v1_text/<task>/`):
  - Image_0.<ext> ... Image_N.<ext>          (references in canonical order,
                                              Image_N = LAYOUT PREVIEW)
  - layout.json                              (has `images` meta — used to
                                              load references in exact order)
  - instruction.txt                          (text prompt given to the generator)
  - generated_nano_banana_pro.png            (optional)
  - generated_gpt_image_1_5.png              (optional)

Per generated image + judge model, this script writes
`judge_<generator>_<judge>.txt` into the task dir and aggregates CSV/JSON
under `judge_results/`.

Criteria (all 1-10):
  1. Text Instruction Following     (non-visual prompt adherence)
  2. Reference Consistency          (subject fidelity)
  3. Vision Instruction Adherence   (STRICT — spatial layout)
  4. Visual Instruction Residue     (STRICT — leftover marks)
  5. Scene Coherence                (foreground-background integration)
  6. Visual Quality                 (overall perceptual quality)
"""
import argparse
import base64
import json
import mimetypes
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

load_dotenv()

DEFAULT_VERSION = "v1"  # use --version to switch (e.g. --version v2)
TASK_ROOT = Path(f"Main/IF-Banana_Task_{DEFAULT_VERSION}_text")  # default; --version overrides
DEFAULT_OUTPUT_DIR = Path("judge_results")
GENERATOR_MODELS = ["nano_banana_pro", "gpt_image_1_5"]
JUDGE_MODELS = ["gemini", "gpt"]
GEMINI_JUDGE_MODEL = "gemini-2.5-flash"
GPT_JUDGE_MODEL = "gpt-5-2025-08-07"

METRIC_NAMES = [
    "Text Instruction Following",
    "Reference Consistency",
    "Vision Instruction Adherence",
    "Visual Instruction Residue",
    "Scene Coherence",
    "Visual Quality",
]
METRIC_SLUGS = {
    "Text Instruction Following":   "text_instruction_following",
    "Reference Consistency":        "reference_consistency",
    "Vision Instruction Adherence": "vision_instruction_adherence",
    "Visual Instruction Residue":   "visual_instruction_residue",
    "Scene Coherence":              "scene_coherence",
    "Visual Quality":               "visual_quality",
}

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

PROMPT_HEADER = """You are a STRICT evaluator for a multi-reference image generation system.

You will see (in this order):
  1. N reference images: Image 0, Image 1, ..., Image (N-1). These are mains
     (subjects), an optional STYLE modifier, and an optional EXTRA (pose /
     clothes / text-style) — in the same order that was shown to the
     generator.
  2. Image N: the LAYOUT PREVIEW — colored bounding boxes on a blank canvas
     that specify WHERE each main subject must be placed in the generated
     image. The color of each box identifies which Image_i belongs in it.
  3. The instruction text that was given to the image generator.
  4. The generated output image.

Reference images follow immediately below (Image 0, then Image 1, ..., then
the LAYOUT PREVIEW as the LAST reference).
"""


def build_middle(instruction):
    return (
        "\n\n=====================================================\n"
        "Instruction text given to the generator:\n"
        "-----------------------------------------------------\n"
        f"{instruction}\n"
        "=====================================================\n\n"
        "Generated image (the output to evaluate):"
    )


PROMPT_CRITERIA = """
Evaluate the generated image on SIX independent criteria, each on a 1-10
scale. Calibration: 4 = average, 6 = good, 8 = excellent, 10 = perfect.

================================================================
1. Text Instruction Following   (non-visual prompt adherence)
================================================================
How well does the generated image follow the natural-language parts of the
instruction text — ignoring the spatial layout (that is criterion 3)?
Check every clause of the instruction:
  - STYLE clause ("Apply Image X's <tone|lighting|style> strictly ..."):
      is that attribute visibly applied across the whole image?
  - BACKGROUND phrase ("in <setting>"):
      is the described setting actually rendered?
  - EXTRA clause ("use Image X as a <pose|clothes|style> reference for
      Image Y"):
      is the modifier applied to the correct target subject and in the
      stated role (pose vs clothes vs style)?
  - other explicit directives (e.g. "seamless", "erase any visual
      instructions", "only the text from Image X without its background").

If a clause is clearly ignored or violated, the score MUST NOT exceed 5.
If the STYLE attribute word is clearly not applied, or if the EXTRA is
applied to the wrong subject, the score MUST NOT exceed 4.
If "only the text without its background" is violated (the reference
paper / panel was copied along with the glyphs), the score MUST NOT exceed 5.

================================================================
2. Reference Consistency   (subject fidelity)
================================================================
For each MAIN subject, judge whether the generated image reproduces the
reference faithfully:
  - person  : face identity, clothing details, accessories
  - animal  : species, breed, coat pattern, markings
  - object  : shape, proportions, surface details
  - text    : EXACT glyphs and font style — NOT the background or paper
              it sat on in the reference

If any one subject fails to match the reference in recognizable detail,
the score MUST NOT exceed 6. If a referenced subject is missing or
replaced with a different subject, the score MUST be 1.

================================================================
3. Vision Instruction Adherence   (STRICT — spatial layout)
================================================================
Evaluate how precisely each MAIN subject is placed inside its assigned
colored bounding box from the LAYOUT PREVIEW (Image N). The color of the
box identifies which Image_i should occupy that region.

Be VERY STRICT. Use this calibration:
  10  : every main subject fills its assigned box tightly, with correct
        relative sizes and positions, and NO part of any subject leaks
        outside its box.
  7-9 : minor deviations (e.g. sub-10% scale mismatch, tiny offset) but
        NO subject protrudes outside its box and NO subject is misplaced.
  4-6 : at least one subject visibly protrudes / leaks outside its
        assigned box (even if the subject is roughly in the right region).
  1-3 : one or more subjects are clearly misplaced, wrong size, or the
        relative positions disagree with the layout.
  1   : any color->subject mapping is wrong (a subject sits in the wrong
        colored box, i.e. subjects are swapped), or the spatial layout
        is broadly ignored.

Hard caps (STRICT — apply even if other aspects look good):
  - If a subject sits in the WRONG colored box (wrong color->subject
    mapping), the score MUST be 1.
  - If any subject is CLEARLY misplaced (wrong region, wrong size, or
    relative positions disagree with the layout), the score MUST NOT
    exceed 3.
  - If any subject visibly PROTRUDES / LEAKS outside its assigned box
    (even partially), the score MUST NOT exceed 6.
  - "Roughly in the right quadrant" is NOT enough — this criterion
    measures precise adherence.

================================================================
4. Visual Instruction Residue   (STRICT — leftover marks)
================================================================
Does the generated image contain any leftover artifacts from the
LAYOUT PREVIEW, OR any newly drawn rectangular frames that the model
added to satisfy the placement instruction literally? Look for:
  - colored bounding box outlines from the LAYOUT PREVIEW
  - ANY colored rectangular frame / border / panel that the model drew
    around a subject (this is a hack where the model literally draws a
    "box" to host the subject, turning the output into a framed collage
    — such a frame still counts as residue even if it is not a direct
    copy of the LAYOUT PREVIEW)
  - arrows, labels, index numbers, markings
  - any visual trace of the layout diagram or scrapbook-style framing

Be STRICT. Calibration:
  10  : perfectly clean, no trace of any visual instruction or
        model-drawn frame.
  7-9 : faint residual color tint or a very subtle edge, NOT a clearly
        recognizable outline or frame.
  5-6 : a recognizable residual / drawn box outline exists but is
        partial / broken / faint.
  4   : exactly ONE complete (or near-complete) box outline or frame
        remains.
  1-3 : MULTIPLE complete box outlines / frames remain, or obvious
        arrows / labels / markings are still clearly visible.

Hard caps (STRICT — apply even if other aspects look good):
  - If ANY visually recognizable box outline OR model-drawn subject frame
    is present, the score MUST be <= 6.
  - If exactly ONE complete box outline / frame is fully preserved or
    drawn, the score MUST be 4.
  - If MULTIPLE complete box outlines / frames remain, the score MUST be 1.
  - A "model-drawn frame" means the output reads like a framed panel
    around a subject — a visible rectangular border, picture-frame, or
    coloured swatch enclosing the subject — not a natural part of the
    scene. Count this the same as a leftover LAYOUT PREVIEW box.

================================================================
5. Scene Coherence   (foreground-background integration)
================================================================
Does the result read as ONE coherent, natural scene?
  - foreground subjects and background blend with consistent lighting,
    perspective, and ground plane
  - the STYLE reference (if any) is applied uniformly across the image
  - the BACKGROUND described in the instruction is actually rendered
  - no collage / split-panel / cut-and-paste appearance
  - the SCENE makes semantic sense as a whole (subjects and background
    belong together in one plausible situation, not merely smoothly
    rendered but semantically unrelated pieces)

Watch specifically for a KNOWN HACK: the model produces a pixel-smooth
composite where each subject sits in its OWN scene context (e.g. subject A
in a park, subject B on a beach, subject C in a studio) and the boundaries
are just blurred. This is NOT coherence — it is a semantic mash-up hidden
by a smooth blend. Detect mismatched:
  - locations / environments (park vs ocean vs indoor)
  - time of day / weather
  - activity or situation (sports vs sleeping vs formal event)
  - light direction or color temperature
across subjects and/or background.

Hard caps (STRICT):
  - Collage or obvious cut-and-paste appearance MUST score <= 3.
  - Any visible stylistic mismatch (lighting / color / tone) between
    subjects and background caps the score at 5.
  - If the scene LOOKS smoothly rendered but is actually a semantically
    incoherent mash-up (different subjects come from clearly different
    situations / contexts, even if the pixels blend) the score MUST NOT
    exceed 3. This applies even if the blending is technically flawless —
    semantic mismatch is considered as severe as collage.

================================================================
6. Visual Quality
================================================================
Overall perceptual quality: resolution, texture, aesthetic composition.
Evaluate independently; do NOT cap based on the other scores.

================================================================
Output format
================================================================
First explain your reasoning, starting the line with 'Reasoning: '.
Then emit the final assessment EXACTLY in this format (one per line):

Text Instruction Following: <1-10>.
Reference Consistency: <1-10>.
Vision Instruction Adherence: <1-10>.
Visual Instruction Residue: <1-10>.
Scene Coherence: <1-10>.
Visual Quality: <1-10>.
"""


# --------------------------------------------------------------------------
# Task loading
# --------------------------------------------------------------------------

def load_task(task_dir):
    layout_path = task_dir / "layout.json"
    instruction_path = task_dir / "instruction.txt"
    if not layout_path.exists():
        raise FileNotFoundError(f"missing {layout_path}")
    if not instruction_path.exists():
        raise FileNotFoundError(f"missing {instruction_path}")
    layout = json.loads(layout_path.read_text())
    instruction = instruction_path.read_text().strip()
    reference_paths = [task_dir / e["file"] for e in layout["images"]]
    for p in reference_paths:
        if not p.exists():
            raise FileNotFoundError(f"missing referenced image: {p}")
    return reference_paths, instruction


def _encode_data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    b64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# --------------------------------------------------------------------------
# Judges
# --------------------------------------------------------------------------

def judge_with_gemini(reference_paths, instruction, generated_path):
    if gemini_client is None:
        raise RuntimeError("GEMINI_API_KEY is not set (check .env)")
    ref_images = [Image.open(p) for p in reference_paths]
    gen_image = Image.open(generated_path)
    contents = [PROMPT_HEADER] + ref_images + [build_middle(instruction), gen_image, PROMPT_CRITERIA]
    response = gemini_client.models.generate_content(
        model=GEMINI_JUDGE_MODEL,
        contents=contents,
    )
    return "".join(p.text or "" for p in response.candidates[0].content.parts)


def judge_with_gpt(reference_paths, instruction, generated_path):
    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY is not set (check .env)")
    content = [{"type": "text", "text": PROMPT_HEADER}]
    for p in reference_paths:
        content.append({"type": "image_url", "image_url": {"url": _encode_data_url(p)}})
    content.append({"type": "text", "text": build_middle(instruction)})
    content.append({"type": "image_url", "image_url": {"url": _encode_data_url(generated_path)}})
    content.append({"type": "text", "text": PROMPT_CRITERIA})
    response = openai_client.chat.completions.create(
        model=GPT_JUDGE_MODEL,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=8192,
    )
    return response.choices[0].message.content


JUDGES = {"gemini": judge_with_gemini, "gpt": judge_with_gpt}


# --------------------------------------------------------------------------
# Per-task worker
# --------------------------------------------------------------------------

def judge_one(task_dir, generator_model, judge_model, overwrite=False):
    generated_path = task_dir / f"generated_{generator_model}.png"
    if not generated_path.exists():
        return task_dir.name, generator_model, "skipped (no generation)"

    judge_path = task_dir / f"judge_{generator_model}_{judge_model}.txt"
    if judge_path.exists() and not overwrite:
        return task_dir.name, generator_model, "skipped (already judged)"

    reference_paths, instruction = load_task(task_dir)
    result = JUDGES[judge_model](reference_paths, instruction, generated_path)
    judge_path.write_text(result)
    time.sleep(0.2)
    return task_dir.name, generator_model, "ok"


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def extract_scores(content):
    scores = {}
    for metric in METRIC_NAMES:
        pattern = rf"{re.escape(metric)}:\s*(\d+)"
        m = re.search(pattern, content)
        if m:
            scores[metric] = int(m.group(1))
    return scores if len(scores) == len(METRIC_NAMES) else None


def extract_results(base_dir, generator_model, judge_model, output_dir):
    per_task = defaultdict(lambda: {m: [] for m in METRIC_NAMES})
    for fp in base_dir.rglob(f"judge_{generator_model}_{judge_model}.txt"):
        try:
            s = extract_scores(fp.read_text())
        except Exception:
            s = None
        if s:
            task = fp.parent.name
            for m, v in s.items():
                per_task[task][m].append(v)

    summary = []
    for task in sorted(per_task.keys()):
        metrics = per_task[task]
        if not metrics[METRIC_NAMES[0]]:
            continue
        row = {"Task": task, "Samples": len(metrics[METRIC_NAMES[0]])}
        for m in METRIC_NAMES:
            vs = metrics[m]
            row[METRIC_SLUGS[m]] = sum(vs) / len(vs) if vs else 0.0
        row["Average"] = sum(row[METRIC_SLUGS[m]] for m in METRIC_NAMES) / len(METRIC_NAMES)
        summary.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"results_{generator_model}_{judge_model}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        header = "Task,Samples," + ",".join(METRIC_SLUGS[m] for m in METRIC_NAMES) + ",Average\n"
        f.write(header)
        for row in summary:
            vals = [f"{row[METRIC_SLUGS[m]]:.3f}" for m in METRIC_NAMES]
            f.write(f"{row['Task']},{row['Samples']}," + ",".join(vals) + f",{row['Average']:.3f}\n")

    json_path = output_dir / f"results_{generator_model}_{judge_model}.json"
    json_out = {}
    for row in summary:
        json_out[row["Task"]] = {
            "samples": row["Samples"],
            **{METRIC_SLUGS[m]: round(row[METRIC_SLUGS[m]], 3) for m in METRIC_NAMES},
            "average": round(row["Average"], 3),
        }
    json_path.write_text(json.dumps(json_out, indent=2, ensure_ascii=False))
    print(f"[{generator_model} x {judge_model}] {len(summary)} tasks -> {csv_path}, {json_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True, choices=JUDGE_MODELS,
                        help="which model judges the output")
    parser.add_argument("--generator", default=None, choices=GENERATOR_MODELS,
                        help="evaluate only this generator; default = all present")
    parser.add_argument("--task", default=None, help="single task dir name")
    parser.add_argument("--version", default=DEFAULT_VERSION,
                        help=f"task pool version (default: {DEFAULT_VERSION}); "
                             "ignored if --base_dir is given")
    parser.add_argument("--base_dir", default=None,
                        help="explicit base dir (overrides --version)")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base = Path(args.base_dir) if args.base_dir else Path(f"Main/IF-Banana_Task_{args.version}_text")
    if args.task is not None:
        task_dirs = [base / args.task]
    else:
        task_dirs = [d for d in sorted(base.iterdir()) if d.is_dir() and not d.name.startswith(".")]

    generators = [args.generator] if args.generator else GENERATOR_MODELS

    work = []
    for d in task_dirs:
        for g in generators:
            if (d / f"generated_{g}.png").exists():
                jf = d / f"judge_{g}_{args.judge}.txt"
                if args.overwrite or not jf.exists():
                    work.append((d, g))

    if not work:
        print("Nothing to judge (all present or no generations found).")
    else:
        print(f"Judging {len(work)} (task, generator) pairs with judge={args.judge}")
        start = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(judge_one, d, g, args.judge, args.overwrite): (d, g)
                       for d, g in work}
            with tqdm(total=len(work), desc=f"judge[{args.judge}]") as pbar:
                for fut in as_completed(futures):
                    name, gen, status = fut.result()
                    tqdm.write(f"[{status}] {name} / {gen}")
                    pbar.update(1)
        elapsed = time.time() - start
        print(f"Completed in {elapsed:.1f}s")

    print("\nAggregating results...")
    for g in generators:
        extract_results(base, g, args.judge, Path(args.output_dir))


if __name__ == "__main__":
    main()
