"""
evaluator_core.py
─────────────────
Cleaned-up OCR + answer-evaluation pipeline.
Original: Google Colab notebook (copy_of_untitled4.py)
Converted to a reusable module for the FastAPI backend.
"""

import cv2
import numpy as np
import pytesseract
import easyocr
import difflib
import re
import os
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ─────────────────────────────────────────────
# GLOBAL MODEL LOADING  (runs once on import)
# ─────────────────────────────────────────────

print("[evaluator_core] Loading EasyOCR reader …")
easyocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

print("[evaluator_core] Loading SentenceTransformer …")
semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("[evaluator_core] Loading TrOCR (microsoft/trocr-large-handwritten) …")
device = "cuda" if torch.cuda.is_available() else "cpu"
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-large-handwritten"
).to(device)

print("[evaluator_core] All models loaded ✓")


# ─────────────────────────────────────────────
# TEXT HELPERS
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


STOPWORDS = {
    "the", "is", "are", "am", "a", "an", "of", "for", "and", "to", "in",
    "on", "at", "this", "that", "these", "those", "it", "as", "by", "with",
    "from", "be", "or", "was", "were", "has", "have", "had",
}


def tokenize(text: str):
    return clean_text(text).split()


def get_keywords(text: str):
    tokens = tokenize(text)
    return set(t for t in tokens if t not in STOPWORDS)


# ─────────────────────────────────────────────
# IMAGE PRE-PROCESSING
# ─────────────────────────────────────────────

def preprocess_image_for_ocr(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10,
    )
    return img, thresh


def upscale_image(image_path: str, scale: float = 2.0, out_path: str = "tmp_big.png"):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    img_big = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(out_path, img_big)
    return out_path


# ─────────────────────────────────────────────
# LINE SEGMENTATION  (for TrOCR)
# ─────────────────────────────────────────────

def segment_into_lines(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = thr.shape
    x1 = int(0.15 * w)
    x2 = int(0.95 * w)
    band = thr[:, x1:x2]

    row_sum = np.sum(band > 0, axis=1)
    ink_threshold = 0.01 * (x2 - x1)

    line_regions = []
    in_line = False
    start_y = 0

    for y in range(h):
        if row_sum[y] > ink_threshold:
            if not in_line:
                in_line = True
                start_y = y
        else:
            if in_line:
                end_y = y
                if end_y - start_y > 12:
                    pad = 3
                    y0 = max(0, start_y - pad)
                    y1 = min(h, end_y + pad)
                    line_regions.append((y0, y1))
                in_line = False

    if in_line:
        end_y = h - 1
        if end_y - start_y > 12:
            pad = 3
            y0 = max(0, start_y - pad)
            y1 = h
            line_regions.append((y0, y1))

    if not line_regions:
        return []

    line_regions = sorted(line_regions, key=lambda t: t[0])

    # Filter tiny / blank regions
    heights = [y1 - y0 for (y0, y1) in line_regions]
    median_h = np.median(heights)

    filtered = []
    for y0, y1 in line_regions:
        if (y1 - y0) < 0.5 * median_h:
            continue
        crop_thr = thr[y0:y1, x1:x2]
        ink_ratio = np.count_nonzero(crop_thr) / float(crop_thr.size)
        if ink_ratio < 0.01:
            continue
        filtered.append((y0, y1))

    if not filtered:
        filtered = line_regions

    return [img[y0:y1, :] for (y0, y1) in filtered]


# ─────────────────────────────────────────────
# OCR ENGINES
# ─────────────────────────────────────────────

def ocr_easyocr(image_path: str) -> str:
    result = easyocr_reader.readtext(image_path, detail=0)
    return " ".join(result)


def ocr_pytesseract(image_path: str) -> str:
    _, thresh = preprocess_image_for_ocr(image_path)
    custom_config = "--psm 6"
    return pytesseract.image_to_string(thresh, config=custom_config)


def ocr_trocr(image_path: str) -> str:
    line_imgs = segment_into_lines(image_path)

    if not line_imgs:
        # Fallback: whole image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        img_resized = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        pil_img = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        pixel_values = trocr_processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values, max_length=512)
        return trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    all_lines = []
    for line_img in line_imgs:
        big = cv2.resize(line_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        pil = Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
        pixel = trocr_processor(images=pil, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            gen = trocr_model.generate(pixel, max_length=128)
        line_text = trocr_processor.batch_decode(gen, skip_special_tokens=True)[0]
        all_lines.append(line_text.strip())

    return "\n".join(all_lines)


# ─────────────────────────────────────────────
# SIMILARITY FUNCTIONS
# ─────────────────────────────────────────────

def keyword_overlap_score(student_text: str, reference_text: str) -> float:
    ref_kw = get_keywords(reference_text)
    stu_kw = get_keywords(student_text)
    if not ref_kw:
        return 0.0
    return len(ref_kw & stu_kw) / len(ref_kw)


def sequence_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, clean_text(a), clean_text(b)).ratio()


def semantic_similarity(a: str, b: str) -> float:
    embs = semantic_model.encode([clean_text(a), clean_text(b)], convert_to_tensor=True)
    sim = util.cos_sim(embs[0], embs[1]).item()
    return max(0.0, min(1.0, sim))


# ─────────────────────────────────────────────
# MAIN EVALUATION FUNCTION
# ─────────────────────────────────────────────

def evaluate_answer_from_image(
    image_path: str,
    reference_answer: str,
    max_marks: float = 5.0,
    verbose: bool = False,
) -> dict:
    """
    Run 3 OCR engines → pick best → compute metrics → return marks dict.
    """
    # Pre-process: upscale
    big_path = upscale_image(image_path, scale=2.0, out_path=image_path + "_big.png")

    ocr_results = {}

    # TrOCR
    try:
        ocr_results["trocr"] = ocr_trocr(big_path)
    except Exception as e:
        print(f"TrOCR failed: {e}")
        ocr_results["trocr"] = ""

    # EasyOCR
    try:
        ocr_results["easyocr"] = ocr_easyocr(big_path)
    except Exception as e:
        print(f"EasyOCR failed: {e}")
        ocr_results["easyocr"] = ""

    # Tesseract
    try:
        ocr_results["tesseract"] = ocr_pytesseract(big_path)
    except Exception as e:
        print(f"Tesseract failed: {e}")
        ocr_results["tesseract"] = ""

    # Pick best engine by semantic similarity to reference
    ocr_scores = {}
    for eng, txt in ocr_results.items():
        ocr_scores[eng] = semantic_similarity(txt, reference_answer) if txt.strip() else 0.0

    best_engine = max(ocr_scores, key=ocr_scores.get)
    best_text = clean_text(ocr_results[best_engine])

    # Detailed metrics
    sem_sim = semantic_similarity(best_text, reference_answer)
    seq_sim = sequence_similarity(best_text, reference_answer)
    key_sim = keyword_overlap_score(best_text, reference_answer)

    # Weighted combination
    final_score = 0.6 * sem_sim + 0.2 * key_sim + 0.2 * seq_sim
    final_score = max(0.0, min(1.0, final_score))
    obtained = round(final_score * max_marks, 2)

    if verbose:
        print(f"  OCR scores: {ocr_scores}")
        print(f"  Best engine: {best_engine}")
        print(f"  Sem={sem_sim:.3f}  Seq={seq_sim:.3f}  Key={key_sim:.3f}")
        print(f"  → {obtained}/{max_marks}")

    # Clean up temp file
    try:
        os.remove(big_path)
    except OSError:
        pass

    return {
        "best_ocr_engine": best_engine,
        "ocr_results": ocr_results,
        "ocr_semantic_scores": ocr_scores,
        "selected_text_clean": best_text,
        "semantic_similarity": sem_sim,
        "sequence_similarity": seq_sim,
        "keyword_coverage": key_sim,
        "final_score_0_1": final_score,
        "marks_obtained": obtained,
        "max_marks": max_marks,
    }


# ─────────────────────────────────────────────
# MARK OVERLAY ON IMAGE
# ─────────────────────────────────────────────

def create_mark_overlay(input_image_path: str, result: dict, output_path: str = "marked.png"):
    img = cv2.imread(input_image_path)
    if img is None:
        raise ValueError("Could not load student answer image.")

    img_out = img.copy()
    mark_text = str(int(round(result["marks_obtained"])))

    h, w = img_out.shape[:2]
    x = int(0.03 * w)
    y = int(0.12 * h)

    cv2.putText(
        img_out, mark_text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 3.5,
        (0, 0, 255), 8, cv2.LINE_AA,
    )
    cv2.imwrite(output_path, img_out)
    return output_path


# ─────────────────────────────────────────────
# TEACHER REPORT  (text-based)
# ─────────────────────────────────────────────

def generate_teacher_report(result: dict, reference_answer: str) -> str:
    best_engine = result["best_ocr_engine"]
    ocr_scores = result["ocr_semantic_scores"]
    sem_sim = result["semantic_similarity"]
    seq_sim = result["sequence_similarity"]
    key_sim = result["keyword_coverage"]
    final_score = result["final_score_0_1"]
    marks_obtained = result["marks_obtained"]
    max_marks = result["max_marks"]
    student_text_raw = result["ocr_results"][best_engine]

    ref_kw = get_keywords(reference_answer)
    stu_kw = get_keywords(student_text_raw)
    covered = sorted(ref_kw & stu_kw)
    missing = sorted(ref_kw - stu_kw)

    lines = []
    lines.append(f"Final Marks: {marks_obtained:.2f} / {max_marks} ({final_score*100:.1f}%)")
    lines.append(f"OCR Engine: {best_engine}")
    lines.append(f"Semantic: {sem_sim:.3f}  |  Sequence: {seq_sim:.3f}  |  Keywords: {key_sim:.3f}")
    lines.append(f"Covered concepts: {', '.join(covered) if covered else 'none detected'}")
    lines.append(f"Missing concepts: {', '.join(missing[:10]) if missing else 'none — all present'}")

    return "\n".join(lines)
