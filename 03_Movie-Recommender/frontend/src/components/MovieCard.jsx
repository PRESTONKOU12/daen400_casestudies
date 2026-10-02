import React from "react";
import StarRating from "./StarRating";

export default function MovieCard({ movie, rating, onRate }) {
  return (
    <div className={`movie-card ${rating > 0 ? "rated" : ""}`}>
      <div>
        <div className="movie-title">{movie.title}</div>
        {movie.genres && <div className="movie-genres">{movie.genres}</div>}
        {movie.count != null && (
          <div className="movie-count">{movie.count.toLocaleString()} ratings</div>
        )}
      </div>
      <StarRating value={rating} onChange={onRate} />
    </div>
  );
}