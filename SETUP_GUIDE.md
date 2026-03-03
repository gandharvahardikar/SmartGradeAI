# Connecting the Answer Evaluator: Complete VS Code Setup Guide

## Architecture Overview

```
┌─────────────────────┐       HTTP        ┌──────────────────────────┐
│   React Frontend    │  ───────────────▶  │   FastAPI Backend         │
│   (localhost:3000)  │  POST /evaluate    │   (localhost:8000)        │
│                     │  ◀───────────────  │                          │
│   answer-evaluator  │   JSON response    │   evaluator_core.py      │
│   -master/          │                    │   (TrOCR + EasyOCR +     │
│                     │  GET /download/..  │    Tesseract + Semantic)  │
└─────────────────────┘                    └──────────────────────────┘
```

The Python Colab notebook becomes the **backend API** (FastAPI).
The React app becomes the **frontend** that talks to it.

---

## Prerequisites

| Tool       | Version   | Install                                           |
|------------|-----------|---------------------------------------------------|
| Python     | 3.10+     | https://www.python.org/downloads/                 |
| Node.js    | 18+       | https://nodejs.org/                                |
| Tesseract  | 5.x       | See OS-specific instructions below                 |
| Git        | any       | https://git-scm.com/                               |
| VS Code    | latest    | https://code.visualstudio.com/                     |
| GPU (optional) | CUDA 11.8+ | Dramatically speeds up TrOCR & EasyOCR       |

### Install Tesseract OCR

**Windows:**
1. Download installer from https://github.com/UB-Mannheim/tesseract/wiki
2. Run installer → note the install path (default: `C:\Program Files\Tesseract-OCR`)
3. Add to PATH: System Properties → Environment Variables → Path → Add the folder
4. Verify: open new terminal → `tesseract --version`

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

---

## Folder Structure (What You'll End Up With)

```
answer-evaluator-project/
├── backend/
│   ├── app.py                  ← FastAPI server
│   ├── evaluator_core.py       ← ML pipeline (cleaned Colab code)
│   ├── requirements.txt        ← Python dependencies
│   ├── outputs/                ← (auto-created) marked PDFs stored here
│   └── venv/                   ← (you create) Python virtual env
│
└── frontend/
    └── answer-evaluator-master/
        ├── package.json
        ├── src/
        │   ├── App.js          ← ★ REPLACED with updated version
        │   ├── index.js
        │   └── index.css
        └── ...
```

---

## Step-by-Step Setup

### STEP 1 — Create the project folder

Open a terminal (or VS Code terminal) and run:

```bash
mkdir answer-evaluator-project
cd answer-evaluator-project
```

### STEP 2 — Set up the Backend

```bash
mkdir backend
```

Copy the three backend files into `backend/`:
- `app.py`
- `evaluator_core.py`
- `requirements.txt`

(These are provided in this download.)

Now create a Python virtual environment and install dependencies:

**Windows (PowerShell):**
```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **⚠️ PyTorch with GPU:** If you have an NVIDIA GPU, install the CUDA version
> of PyTorch FIRST for much faster processing:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```
> Then install the rest: `pip install -r requirements.txt`

> **First run will download ~1.5 GB of ML models** (TrOCR, SentenceTransformer).
> This only happens once; they're cached in `~/.cache/huggingface/`.

### STEP 3 — Set up the Frontend

Go back to the project root:
```bash
cd ..
```

Unzip `answer-evaluator-master.zip` into a `frontend/` folder:
```bash
mkdir frontend
# Unzip into frontend/ so the path is frontend/answer-evaluator-master/
```

Or if you already have the extracted folder, just move it:
```bash
mv answer-evaluator-master frontend/
```

**Replace `App.js`** with the updated version:
```bash
# Copy the provided App.js over the old one
cp App.js frontend/answer-evaluator-master/src/App.js
```

(The updated `App.js` is provided in this download.)

Install Node dependencies:
```bash
cd frontend/answer-evaluator-master
npm install
```

### STEP 4 — Verify the `App.js` API URL

Open `frontend/answer-evaluator-master/src/App.js` and confirm this line:

```javascript
const API_URL = "http://localhost:8000";
```

This must match the port the backend runs on (default: 8000).

---

## Running the Application

You need **two terminals** running simultaneously.

### Terminal 1 — Start Backend (Python)

```bash
cd answer-evaluator-project/backend

# Activate virtual env
# Windows:  .\venv\Scripts\Activate.ps1
# Mac/Linux: source venv/bin/activate

python app.py
```

You should see:
```
[evaluator_core] Loading EasyOCR reader …
[evaluator_core] Loading SentenceTransformer …
[evaluator_core] Loading TrOCR (microsoft/trocr-large-handwritten) …
[evaluator_core] All models loaded ✓
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test it:** Open http://localhost:8000/health in your browser → should show `{"status":"ok"}`

### Terminal 2 — Start Frontend (React)

```bash
cd answer-evaluator-project/frontend/answer-evaluator-master
npm start
```

This opens http://localhost:3000 in your browser automatically.

---

## Using the Application

1. Open http://localhost:3000
2. Upload a **Student Answer Sheet PDF** (scanned handwritten answers)
3. Upload a **Reference Answer PDF** (typed model answers)
4. Set the max marks
5. Click **Evaluate**
6. Wait (first evaluation is slow due to model warm-up; ~30-60 sec per page)
7. View per-page marks, similarity scores, extracted text, and teacher reports
8. Download the marked PDF with red score overlays

---

## VS Code Setup Tips

### Recommended Extensions
- **Python** (ms-python.python) — backend IntelliSense
- **Pylance** (ms-python.vscode-pylance) — type checking
- **ES7+ React/Redux** (dsznajder.es7-react-js-snippets) — frontend snippets
- **REST Client** (humao.rest-client) — test API endpoints

### Open as Workspace

1. File → Open Folder → select `answer-evaluator-project/`
2. You'll see both `backend/` and `frontend/` in the explorer

### Set Python Interpreter

1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Choose `./backend/venv/bin/python` (or `.\backend\venv\Scripts\python.exe` on Windows)

### Run Both with VS Code Tasks

Create `.vscode/tasks.json` in your project root:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Backend",
      "type": "shell",
      "command": "${workspaceFolder}/backend/venv/bin/python",
      "windows": {
        "command": "${workspaceFolder}\\backend\\venv\\Scripts\\python.exe"
      },
      "args": ["app.py"],
      "options": { "cwd": "${workspaceFolder}/backend" },
      "isBackground": true,
      "problemMatcher": []
    },
    {
      "label": "Frontend",
      "type": "shell",
      "command": "npm",
      "args": ["start"],
      "options": { "cwd": "${workspaceFolder}/frontend/answer-evaluator-master" },
      "isBackground": true,
      "problemMatcher": []
    },
    {
      "label": "Start All",
      "dependsOn": ["Backend", "Frontend"],
      "problemMatcher": []
    }
  ]
}
```

Then: `Ctrl+Shift+P` → "Tasks: Run Task" → "Start All"

---

## Troubleshooting

### "CORS error" in browser console
Make sure the backend is running on port 8000 and `API_URL` in App.js is `http://localhost:8000`.

### "Tesseract not found"
**Windows:** Add Tesseract install folder to your system PATH.
**Mac/Linux:** Run `which tesseract` — if empty, install it (see Prerequisites).

You can also set it explicitly in `evaluator_core.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### "CUDA out of memory"
TrOCR-large needs ~2-3 GB VRAM. If you're running out:
- Close other GPU apps
- Or switch to CPU by editing `evaluator_core.py`: change `device = "cpu"`
- Or use a smaller model: replace `trocr-large-handwritten` with `trocr-base-handwritten`

### Backend starts but evaluation fails
- Check the terminal running `python app.py` for the full traceback
- Make sure the reference PDF contains **selectable text** (not a scanned image)
- Make sure the student PDF has at least one page

### "Module not found" errors
Make sure your virtual environment is activated before running `python app.py`.

### Very slow evaluation
- First run downloads models (~1.5 GB) — subsequent runs are much faster
- Without GPU: ~60-120 sec per page (TrOCR is heavy)
- With GPU: ~15-30 sec per page
- You can disable TrOCR for faster (but less accurate) results by commenting out the TrOCR block in `evaluator_core.py`

---

## Key Differences from the Original Colab Notebook

| Colab Version | Backend Version |
|---------------|-----------------|
| `!pip install ...` | `requirements.txt` + venv |
| `from google.colab import files` | FastAPI file upload endpoint |
| `files.upload()` | `UploadFile` parameter |
| `plt.show()` | Removed (no GUI on server) |
| Hardcoded reference answer | Extracted from uploaded reference PDF |
| Single image input | PDF input → auto-converted to page images |
| Print output | JSON API response |
| `matplotlib` visualizations | Per-page metrics in React frontend |
| ngrok tunnel | localhost (or deploy to any server) |

---

## Optional: Deploy for Remote Access

If you need others to access (like the original ngrok setup):

**Option A — ngrok (quick)**
```bash
ngrok http 8000
```
Then update `API_URL` in `App.js` to the ngrok URL.

**Option B — Deploy backend to a cloud server**
Deploy the `backend/` folder to any server (AWS EC2, Google Cloud, Railway, Render, etc.)
and update `API_URL` in the React app accordingly.
