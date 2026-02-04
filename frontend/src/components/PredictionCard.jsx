function PredictionCard({ result, preview }) {
  return (
    <div
      style={{
        marginTop: "20px",
        padding: "15px",
        border: "1px solid #ddd",
        borderRadius: "8px",
      }}
    >
     

      <h3 style={{ marginTop: "15px" }}>Prediction Result</h3>

      <p>
        <strong>Class:</strong> {result.predicted_label}
      </p>
      <p>
        <strong>Confidence:</strong> {result.confidence}
      </p>
    </div>
  );
}

export default PredictionCard;
