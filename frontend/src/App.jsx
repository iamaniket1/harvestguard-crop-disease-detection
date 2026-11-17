import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const inputRef = useRef(null);

  useEffect(() => {
    if (!file) {
      setPreview("");
      return undefined;
    }

    const previewUrl = URL.createObjectURL(file);
    setPreview(previewUrl);

    return () => URL.revokeObjectURL(previewUrl);
  }, [file]);

  function chooseImage(selectedFile) {
    if (!selectedFile) return;

    setError("");
    setResult(null);

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please choose an image file.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Please choose an image smaller than 10 MB.");
      return;
    }

    setFile(selectedFile);
  }

  async function analyzeLeaf() {
    if (!file || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Could not analyze this image."
        );
      }

      setResult(data);
    } catch (requestError) {
      if (requestError instanceof TypeError) {
        setError(
          "Cannot reach the prediction server. Start the backend first."
        );
      } else {
        setError(requestError.message);
      }
    } finally {
      setLoading(false);
    }
  }

  function removeImage() {
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  const topFive = result
    ? result.scores.slice(0, 5)
    : [];

  return (
    <div className="app-shell">
      <header className="navbar">
        <a className="brand" href="#home">
          <span className="brand-icon">✦</span>
          Harvest<span>Guard</span>
        </a>

        <a className="nav-link" href="#about">
          About the model
        </a>
      </header>

      <main id="home">
        <section className="hero">
          <p className="eyebrow">
            <span className="status-dot" />
            CROP LEAF DISEASE CLASSIFICATION
          </p>

          <h1>
            Understand your leaves.
            <span> Protect your harvest.</span>
          </h1>

          <p className="hero-text">
            Upload a crop-leaf image to compare it against
            38 trained PlantVillage disease and healthy classes.
          </p>

          <div className="tag-list">
            <span>38 classes</span>
            <span>14 crop types</span>
            <span>96.67% test accuracy</span>
          </div>
        </section>

        <section className="prediction-workspace">
          <div className="upload-panel">
            <p className="panel-label">
              01 / UPLOAD IMAGE
            </p>

            <h2>Choose a leaf photo</h2>

            <input
              ref={inputRef}
              id="leaf-upload"
              className="hidden-input"
              type="file"
              accept="image/*"
              onChange={(event) => {
                chooseImage(event.target.files?.[0]);
                event.target.value = "";
              }}
            />

            <button
              type="button"
              className="upload-zone"
              onClick={() => inputRef.current?.click()}
            >
              {preview ? (
                <img
                  src={preview}
                  alt="Selected crop leaf"
                  className="image-preview"
                />
              ) : (
                <span className="upload-content">
                  <b>↑</b>
                  <strong>Click to upload a leaf image</strong>
                  <small>
                    JPG, PNG, and similar files · Maximum 10 MB
                  </small>
                </span>
              )}
            </button>

            <div className="file-row">
              <span>
                {file
                  ? file.name
                  : "Use a clear close-up image of one leaf."}
              </span>

              {file && (
                <button
                  type="button"
                  className="remove-button"
                  onClick={removeImage}
                >
                  Remove
                </button>
              )}
            </div>

            <button
              type="button"
              className="analyze-button"
              onClick={analyzeLeaf}
              disabled={!file || loading}
            >
              {loading
                ? "Analyzing image…"
                : "Analyze leaf →"}
            </button>

            {error && (
              <p className="error-message">
                {error}
              </p>
            )}
          </div>

          <div className="result-panel">
            <p className="panel-label">
              02 / MODEL RESULT
            </p>

            {!result && !loading && (
              <div className="empty-result">
                <span>✦</span>
                <h2>Results will appear here</h2>
                <p>
                  Upload an image and click Analyze leaf
                  to see the five strongest predictions.
                </p>
              </div>
            )}

            {loading && (
              <div className="empty-result">
                <div className="spinner" />
                <h2>Looking at leaf patterns</h2>
                <p>
                  The model is analyzing your image.
                </p>
              </div>
            )}

            {result && (
              <div className="result-content">
                <p className="prediction-label">
                  HIGHEST MODEL SCORE
                </p>

                <h2 className="prediction-name">
                  {result.prediction}
                </h2>

                <p className="main-score">
                  {(result.model_score * 100).toFixed(1)}%
                </p>

                <p className="top-five-label">
                  Top five predictions
                </p>

                <div className="score-list">
                  {topFive.map((item) => (
                    <div
                      className="score-row"
                      key={item.class_name}
                    >
                      <div className="score-text">
                        <span>{item.class_name}</span>
                        <strong>
                          {(item.score * 100).toFixed(1)}%
                        </strong>
                      </div>

                      <div className="score-track">
                        <div
                          className="score-fill"
                          style={{
                            width: `${Math.min(
                              item.score * 100,
                              100
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <p className="notice">
                  {result.notice}
                </p>
              </div>
            )}
          </div>
        </section>

        <section className="about-section" id="about">
          <div>
            <p className="panel-label">
              RESEARCH RESULTS
            </p>

            <h2>
              CNN compared with
              Whale Optimization tuning.
            </h2>

            <p>
              The project used WOA to tune learning rate,
              dropout, and image rotation. The baseline
              MobileNetV2 was selected because it had the
              lower validation loss.
            </p>
          </div>

          <div className="metric-grid">
            <div>
              <strong>96.67%</strong>
              <span>Baseline test accuracy</span>
            </div>

            <div>
              <strong>96.43%</strong>
              <span>WOA test accuracy</span>
            </div>

            <div>
              <strong>38</strong>
              <span>Supported classes</span>
            </div>
          </div>
        </section>

        <p className="scope-note">
          Supports only the 38 trained PlantVillage classes.
          Real-world field-photo performance has not been
          established. A model score is not a confirmed diagnosis.
        </p>
      </main>

      <footer>
        <span>HarvestGuard</span>
        <span>Crop disease detection research project</span>
      </footer>
    </div>
  );
}