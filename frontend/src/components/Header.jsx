import logo from "../assets/logo.png";

function Header() {
  return (
    <header
      style={{
        display: "flex",
        alignItems: "center",
        gap: "16px",
        padding: "24px 32px",
        marginBottom: "24px",
      }}
    >
      <img
        src={logo}
        alt="Logo"
        style={{
          width: "56px",
          height: "56px",
          transform: "translateX(-30px)"
        }}
      />

      <div>
        <h1
          style={{
            fontSize: "2.4rem",
            fontWeight: "700",
            margin: 0,
            lineHeight: "1.1",
            letterSpacing: "-0.02em",
            color:"rgb(112, 98, 67)",
            transform: "translateX(-30px)"
          }}
        >
          PawVision 
        </h1>

        <p
          style={{
            margin: "4px 0 0",
            fontSize: "1rem",
            opacity: 0.75,
            transform: "translateX(-20px)"
          }}
        >
          Cat vs Dog Image Classifier
        </p>
      </div>
    </header>
  );
}

export default Header;