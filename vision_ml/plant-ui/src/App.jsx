import { useEffect, useState } from "react";

const API_BASE_URL = "http://172.20.10.14:8000"; // IP Raspberry

const initialSensor = {
  light: 200,
  temperature: 22.5,
  soil_capacitance: 550,
  hours_since_last_watering: 12,
  delta_soil: -2.0,
  rolling_avg_light_6h: 150,
  light_hours_today: 1.0,
  hour_of_day: 12,
};

function App() {
  const [sensorData, setSensorData] = useState(initialSensor); // debug/manual
  const [sensorResult, setSensorResult] = useState(null);
  const [sensorFeatures, setSensorFeatures] = useState(null); // z Influx
  const [visionResult, setVisionResult] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const callSensorApi = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE_URL}/sensor/predict/live`, {
        method: "POST",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setSensorResult(json.sensor_AI_result || json);
      setSensorFeatures(json.features || null);
    } catch (e) {
      setError("Błąd /sensor/predict/live: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  // Auto-refresh
  useEffect(() => {
    callSensorApi(); // pierwszy odczyt
    const interval = setInterval(() => {
      callSensorApi();
    }, 30000); // 30s
    return () => clearInterval(interval);
  }, []);

  const callVisionApi = async () => {
    if (!imageFile) {
      setError("Wybierz zdjęcie rośliny.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", imageFile);
      const res = await fetch(`${API_BASE_URL}/vision/predict`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setVisionResult(json.vision_AI_result || json);
    } catch (e) {
      setError("Błąd /vision/predict: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  const prettyJson = (obj) =>
    obj ? (
      <pre style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>
        {JSON.stringify(obj, null, 2)}
      </pre>
    ) : (
      <span style={{ opacity: 0.7, fontSize: 13 }}>Brak danych</span>
    );

  const renderSensorBadge = (active, label, okText) => {
    const isProblem = Boolean(active);
    return (
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          padding: "0.35rem 0.6rem",
          borderRadius: 6,
          marginBottom: 4,
          fontSize: 13,
          background: isProblem ? "#3b0000" : "#012a10",
          border: `1px solid ${isProblem ? "#ff5252" : "#00e676"}`,
        }}
      >
        <span>{label}</span>
        <strong>{isProblem ? "TAK" : okText ?? "OK"}</strong>
      </div>
    );
  };

  const renderVisionSummary = () => {
    if (!visionResult) {
      return <span style={{ opacity: 0.7, fontSize: 13 }}>Brak analizy.</span>;
    }
    const bucket = visionResult.top_bucket;
    const label = visionResult.top_label;
    const isHealthy = bucket === "healthy";
    const bucketText = isHealthy ? "ZDROWA / W NORMIE" : "POD STRESEM";
    const bucketColor = isHealthy ? "#00e676" : "#ff5252";
    const bucketBg = isHealthy ? "#00351d" : "#3b0000";
    return (
      <div>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            padding: "0.35rem 0.7rem",
            borderRadius: 999,
            background: bucketBg,
            color: bucketColor,
            border: `1px solid ${bucketColor}`,
            marginBottom: 6,
          }}
        >
          <strong>{bucketText}</strong>
        </div>
        <div style={{ fontSize: 13 }}>
          <div><strong>Klasa:</strong> {label}</div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ fontFamily: "sans-serif", padding: "1.5rem", maxWidth: 1100, margin: "0 auto", color: "#eee", background: "#121212", minHeight: "100vh" }}>
      <header style={{ display: "flex", alignItems: "center", marginBottom: 16 }}>
        <h1 style={{ margin: 0 }}>Plant AI Panel</h1>
        <div style={{ marginLeft: 12, fontSize: 13, opacity: 0.8 }}>
          Backend: <code>{API_BASE_URL}</code>
        </div>
      </header>

      {error && (
        <div style={{ background: "#4b0000", padding: "0.75rem 1rem", borderRadius: 8, marginBottom: "1rem", border: "1px solid #ff5252" }}>
          {error}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
        {/* SENSOR */}
        <section style={{ background: "#1e1e1e", padding: "1rem", borderRadius: 8, border: "1px solid #333" }}>
          <h2>Sensory (LIVE)</h2>

          <button
            onClick={callSensorApi}
            disabled={loading}
            style={{ marginBottom: 10, padding: "0.5rem 1rem", borderRadius: 6, border: "none", background: "#00c853", color: "#000", fontWeight: 600 }}
          >
            {loading ? "Czekaj..." : "Pobierz LIVE (/sensor/predict/live)"}
          </button>

          {sensorResult ? (
            <>
              {renderSensorBadge(sensorResult.forgot_to_water, "Zapomniałeś podlać?", "OK")}
              {renderSensorBadge(sensorResult.too_dark_today, "Za ciemno dzisiaj?", "OK")}
              {renderSensorBadge(sensorResult.worth_relocating, "Warto przestawić?", "Nie")}
              {renderSensorBadge(sensorResult.too_bright_now, "Za jasno teraz?", "OK")}
            </>
          ) : (
            <span style={{ opacity: 0.7, fontSize: 13 }}>Brak odczytu.</span>
          )}

          <details style={{ marginTop: 8 }}>
            <summary style={{ cursor: "pointer", fontSize: 13 }}>Debug JSON (sensory)</summary>
            {prettyJson(sensorResult)}
          </details>

          <details style={{ marginTop: 8 }}>
            <summary style={{ cursor: "pointer", fontSize: 13 }}>Debug JSON (features z Influx)</summary>
            {prettyJson(sensorFeatures)}
          </details>
        </section>

        {/* VISION */}
        <section style={{ background: "#1e1e1e", padding: "1rem", borderRadius: 8, border: "1px solid #333" }}>
          <h2>Vision</h2>
          <input type="file" accept="image/*" onChange={(e) => setImageFile(e.target.files?.[0] || null)} />
          <br />
          <button
            onClick={callVisionApi}
            disabled={loading}
            style={{ marginTop: 10, padding: "0.5rem 1rem", borderRadius: 6, border: "none", background: "#00b0ff", color: "#000", fontWeight: 600 }}
          >
            {loading ? "Czekaj..." : "Wyślij do /vision/predict"}
          </button>

          <h3 style={{ marginTop: 10 }}>Interpretacja</h3>
          {renderVisionSummary()}

          <details style={{ marginTop: 8 }}>
            <summary style={{ cursor: "pointer", fontSize: 13 }}>Debug JSON (vision)</summary>
            {prettyJson(visionResult)}
          </details>
        </section>
      </div>
    </div>
  );
}

export default App;
