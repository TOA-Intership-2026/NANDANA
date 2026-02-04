function Button({ text }) {
  return (
    <button
      style={{
        marginTop: "10px",
        padding: "10px",
        width: "100%",
        cursor: "pointer",
      }}
    >
      {text}
    </button>
  );
}

export default Button;
