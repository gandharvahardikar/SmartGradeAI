import { useState } from "react";
import axios from "axios";

function App() {
  const [studentPdf, setStudentPdf] = useState(null);
  const [referencePdf, setReferencePdf] = useState(null);
  const [maxMarks, setMaxMarks] = useState(5);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // ✅ Points to local FastAPI backend
  const API_URL = "http://localhost:8000";

  const submitHandler = async () => {
    if (!studentPdf || !referencePdf) {
      alert("Upload both Student PDF and Reference PDF");
      return;
    }

    const formData = new FormData();
    formData.append("student_pdf", studentPdf);
    formData.append("reference_pdf", referencePdf);
    formData.append("max_marks", maxMarks);

    try {
      setLoading(true);
      setResult(null);

      const res = await axios.post(`${API_URL}/evaluate`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(res.data);
    } catch (err) {
      console.error(err.response?.data || err);
      alert(
        "Evaluation failed: " +
          (err.response?.data?.detail || err.message)
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>SmartGrade AI</h1>
        <p>Upload Student &amp; Reference PDFs to get automatic marks</p>
      </header>

      <div className="main">
        {/* ── Upload Card ── */}
        <div className="card">
          <h2>Upload PDFs</h2>

          <label>Student Answer Sheet (PDF)</label>
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setStudentPdf(e.target.files[0])}
          />

          <label>Reference Answer Sheet (PDF)</label>
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setReferencePdf(e.target.files[0])}
          />

          <label>Max Marks</label>
          <input
            type="number"
            min="1"
            value={maxMarks}
            onChange={(e) => setMaxMarks(Number(e.target.value))}
            style={{
              width: "100%",
              padding: "10px",
              marginTop: "6px",
              borderRadius: "6px",
              border: "1px solid #ccc",
            }}
          />

          <button onClick={submitHandler} disabled={loading}>
            {loading ? "Evaluating …" : "Evaluate"}
          </button>
        </div>

        {/* ── Loader ── */}
        {loading && (
          <div className="card loader-card">
            <div className="spinner"></div>
            <p>Evaluating answer sheet… This may take a few minutes.</p>
          </div>
        )}

        {/* ── Results ── */}
        {result && (
          <div className="card result">
            <h2>Evaluation Result</h2>

            <div className="total-score">
              Total Marks: {result.total_marks}
            </div>

            {result.page_wise_marks.map((p) => (
              <div key={p.page} className="page">
                <h3>Page {p.page}</h3>
                <p>
                  Marks: {p.marks} / {p.max_marks}
                </p>

                <div className="bar">
                  <div
                    className="fill"
                    style={{
                      width: `${(p.marks / p.max_marks) * 100}%`,
                    }}
                  ></div>
                </div>

                <div className="metrics">
                  <span>Semantic: {p.semantic_similarity}</span>
                  <span>Sequence: {p.sequence_similarity}</span>
                  <span>Keywords: {p.keyword_coverage}</span>
                </div>

                <p style={{ fontSize: "13px", color: "#555", marginTop: 6 }}>
                  OCR engine: <strong>{p.best_ocr_engine}</strong>
                </p>

                {p.extracted_text && (
                  <details style={{ marginTop: 8 }}>
                    <summary style={{ cursor: "pointer", color: "#2563eb" }}>
                      View extracted text
                    </summary>
                    <pre
                      style={{
                        background: "#f5f5f5",
                        padding: 10,
                        borderRadius: 6,
                        whiteSpace: "pre-wrap",
                        fontSize: 13,
                        maxHeight: 200,
                        overflow: "auto",
                      }}
                    >
                      {p.extracted_text}
                    </pre>
                  </details>
                )}

                {p.report && (
                  <details style={{ marginTop: 8 }}>
                    <summary style={{ cursor: "pointer", color: "#16a34a" }}>
                      View teacher report
                    </summary>
                    <pre
                      style={{
                        background: "#f0fdf4",
                        padding: 10,
                        borderRadius: 6,
                        whiteSpace: "pre-wrap",
                        fontSize: 13,
                      }}
                    >
                      {p.report}
                    </pre>
                  </details>
                )}
              </div>
            ))}

            {/* Download Marked PDF */}
            <a
              href={`${API_URL}/download/${result.final_marked_pdf}`}
              target="_blank"
              rel="noreferrer"
              className="download-btn"
            >
              View / Download Marked Answer Sheet PDF
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
