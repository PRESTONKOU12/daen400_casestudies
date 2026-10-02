import React from "react";

export default function Recommendations({ items, onBack }) {
  if (!items || items.length === 0) {
    return (
      <div className="recs-panel">
        <div className="recs-header">
          <h2>No recommendations found</h2>
          <button className="back-btn" onClick={onBack}>
            ← Rate more movies
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="recs-panel">
      <div className="recs-header">
        <h2>🍿 Your Recommendations</h2>
        <button className="back-btn" onClick={onBack}>
          ← Start over
        </button>
      </div>

      <div className="rec-list">
        {items.map((item, i) => (
          <div className="rec-item" key={item.movieId}>
            <span className="rec-rank">{i + 1}</span>
            <div className="rec-info">
              <div className="rec-title">{item.title}</div>
              <div className="rec-genres">{item.genres}</div>
              <div className="rec-when">
                📅 <strong>{item.watch_when}</strong> — {item.watch_blurb}
              </div>
            </div>
            <span className="rec-score">
              {item.predicted_rating.toFixed(2)} ★
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}