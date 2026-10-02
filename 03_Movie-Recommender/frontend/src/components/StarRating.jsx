import React, { useState } from "react";

export default function StarRating({ value = 0, onChange }) {
  const [hover, setHover] = useState(0);

  return (
    <div className="star-row">
      {[1, 2, 3, 4, 5].map((star) => (
        <span
          key={star}
          className={`star ${star <= (hover || value) ? "filled" : ""}`}
          onClick={() => onChange(star === value ? 0 : star)}
          onMouseEnter={() => setHover(star)}
          onMouseLeave={() => setHover(0)}
        >
          ★
        </span>
      ))}
      {value > 0 && (
        <button className="clear-btn" onClick={() => onChange(0)}>
          clear
        </button>
      )}
    </div>
  );
}