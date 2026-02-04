import { predictImage } from "../services/api";

function ImageUploader({ file, setFile, preview, setPreview, setResult }) {
  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null); // reset old result
  };

  const handleSubmit = async () => {
    if (!file) return;

    try {
      const response = await predictImage(file);
      setResult(response);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ marginTop: "10px" }}>
      <input type="file" accept="image/*" onChange={handleFileChange} />

      {preview && !file?.predicted && (
        <div style={{ marginTop: "15px" }}>
          <img
            src={preview}
            alt="preview"
            style={{ width: "100%", borderRadius: "8px" }}
          />
        </div>
      )}

      {/* Show submit ONLY before prediction */}
      {file && !preview?.predicted && (
        <button
          onClick={handleSubmit}
          style={{
            marginTop: "15px",
            padding: "50px",
            width: "100%",
            cursor: "pointer",
            color: "rgb(0, 255, 0)"
          }}
        >
          Submit
        </button>
      )}
    </div>
  );
}

export default ImageUploader;
