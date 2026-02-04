import { useState } from "react";
import Header from "../components/Header";
import { predictImage } from "../services/api";
import catImg from "../assets/cattt.png";
import dogImg from "../assets/doggg.png";
import Ballpit from "../components/Ballpit";
import InputFileUpload from "../components/UploadButton";

function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
  };

  const handleSubmit = async () => {
    if (!file) return;

    setLoading(true);
    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (err) {
      alert("Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* 🔹 Background */}
      <Ballpit
        count={50}
        gravity={0.01}
        friction={0.9975}
        wallBounce={1}
        colors={[
          "#5e4932",
          "#000000",
          "#c06464",
          "#ecdd56",
          "#8d8a89",
        ]}
      />

      {/* 🔹 Foreground UI */}
      <div
        style={{
          minHeight: "100vh",
          padding: "24px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          position: "relative",
          zIndex: 1,
        }}
      >
        <Header />

        <div
  style={{
  marginTop: "40px",
  display: "flex",
  justifyContent: "center",
  width: "10%",
  transform: "translateX(0px)", 
}}

>
  <InputFileUpload handleFile={handleFileChange}></InputFileUpload>
</div>

        {preview && (
          <div
            style={{
              width: "100%",
              maxWidth: "420px",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              marginTop: "24px",
            }}
          >
            <img
              src={preview}
              alt="Uploaded"
              style={{
                width: "100%",
                maxHeight: "360px",
                objectFit: "contain",
                borderRadius: "16px",
                boxShadow: "40px"
              }}
            />

            {!result && (
              <button onClick={handleSubmit} style={{ marginTop: "20px" ,color:"rgb(19, 20, 19)",backgroundColor:"rgb(95, 228, 95)", width:"100px",height:"40px",padding:"10px",borderRadius: "10px", }}>
                {loading ? "Predicting..." : "Submit"}
              </button>
            )}

            {result && (
              <div style={{ marginTop: "20px", textAlign: "center" }}>
                <h3>{result.predicted_label}</h3>
                <p>Confidence: {result.confidence}</p>
              </div>
            )}
          </div>
        )}

        <img
          src={catImg}
          alt="Cat"
          style={{
            position: "fixed",
            bottom: 0,
            left: 20,
            width: 350,
            pointerEvents: "none",
          }}
        />

        <img
          src={dogImg}
          alt="Dog"
          style={{
            position: "fixed",
            bottom: 0,
            right: 20,
            width: 350,
            pointerEvents: "none",
          }}
        />
      </div>
    </>
  );
}

export default Home;
