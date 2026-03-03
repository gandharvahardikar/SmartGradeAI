"""
app.py  –  FastAPI backend for the Answer Evaluator
────────────────────────────────────────────────────
Endpoints
  POST /evaluate          multipart: student_pdf, reference_pdf, max_marks
  GET  /download/{name}   serves marked PDF from the outputs folder
  GET  /health            liveness check
"""

import os
import uuid
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# PDF → image conversion
import fitz  # PyMuPDF

# Our ML pipeline
from evaluator_core import (
    evaluate_answer_from_image,
    create_mark_overlay,
    generate_teacher_report,
)

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(title="Answer Evaluator API")

# Allow the React dev server (port 3000) and any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Helper: PDF → list of page images
# ─────────────────────────────────────────────

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[str]:
    """Convert each page of a PDF to a PNG image. Returns list of image paths."""
    doc = fitz.open(pdf_path)
    image_paths = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        zoom = dpi / 72                          # 72 is the default PDF DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_path = pdf_path.replace(".pdf", f"_page{page_num+1}.png")
        pix.save(img_path)
        image_paths.append(img_path)
    doc.close()
    return image_paths


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF (used for the reference answer)."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


# ─────────────────────────────────────────────
# Helper: stitch page images into a single PDF
# ─────────────────────────────────────────────

def images_to_pdf(image_paths: list[str], output_pdf: str):
    """Combine a list of images into one PDF."""
    doc = fitz.open()
    for img_path in image_paths:
        img = fitz.open(img_path)
        rect = img[0].rect
        pdf_page = doc.new_page(width=rect.width, height=rect.height)
        pdf_page.insert_image(rect, filename=img_path)
        img.close()
    doc.save(output_pdf)
    doc.close()


# ─────────────────────────────────────────────
# POST /evaluate
# ─────────────────────────────────────────────

@app.post("/evaluate")
async def evaluate(
    student_pdf: UploadFile = File(...),
    reference_pdf: UploadFile = File(...),
    max_marks: float = Form(5),
):
    """
    Accepts student + reference PDFs.
    Evaluates each page of the student PDF against the reference text.
    Returns JSON that the React frontend expects.
    """
    job_id = uuid.uuid4().hex[:10]
    tmp_dir = tempfile.mkdtemp(prefix=f"eval_{job_id}_")

    try:
        # ── 1. Save uploaded files ──────────────────────────
        stu_path = os.path.join(tmp_dir, "student.pdf")
        ref_path = os.path.join(tmp_dir, "reference.pdf")

        with open(stu_path, "wb") as f:
            f.write(await student_pdf.read())
        with open(ref_path, "wb") as f:
            f.write(await reference_pdf.read())

        # ── 2. Extract reference answer text ────────────────
        reference_text = extract_text_from_pdf(ref_path)
        if not reference_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from the reference PDF. "
                       "Make sure it contains selectable text (not just an image).",
            )

        # ── 3. Convert student PDF pages → images ──────────
        page_images = pdf_to_images(stu_path, dpi=200)
        num_pages = len(page_images)

        if num_pages == 0:
            raise HTTPException(status_code=400, detail="Student PDF has 0 pages.")

        marks_per_page = max_marks / num_pages

        # ── 4. Evaluate each page ──────────────────────────
        page_results = []
        marked_images = []
        total_marks = 0.0

        for i, img_path in enumerate(page_images):
            print(f"\n[Job {job_id}] Evaluating page {i+1}/{num_pages} …")

            result = evaluate_answer_from_image(
                image_path=img_path,
                reference_answer=reference_text,
                max_marks=marks_per_page,
                verbose=True,
            )

            # Create marked overlay
            marked_path = os.path.join(tmp_dir, f"marked_page{i+1}.png")
            create_mark_overlay(img_path, result, output_path=marked_path)
            marked_images.append(marked_path)

            page_marks = result["marks_obtained"]
            total_marks += page_marks

            page_results.append({
                "page": i + 1,
                "marks": round(page_marks, 2),
                "max_marks": round(marks_per_page, 2),
                "semantic_similarity": round(result["semantic_similarity"], 3),
                "sequence_similarity": round(result["sequence_similarity"], 3),
                "keyword_coverage": round(result["keyword_coverage"], 3),
                "best_ocr_engine": result["best_ocr_engine"],
                "extracted_text": result["selected_text_clean"],
                "report": generate_teacher_report(result, reference_text),
            })

        # ── 5. Build final marked PDF ──────────────────────
        final_pdf_name = f"marked_{job_id}.pdf"
        final_pdf_path = os.path.join(OUTPUT_DIR, final_pdf_name)
        images_to_pdf(marked_images, final_pdf_path)

        total_marks = round(total_marks, 2)

        return {
            "total_marks": f"{total_marks} / {max_marks}",
            "total_marks_num": total_marks,
            "max_marks": max_marks,
            "page_wise_marks": page_results,
            "final_marked_pdf": final_pdf_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp dir (keep output PDF)
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────
# GET /download/{filename}
# ─────────────────────────────────────────────

@app.get("/download/{filename}")
async def download(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="application/pdf", filename=filename)


# ─────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
