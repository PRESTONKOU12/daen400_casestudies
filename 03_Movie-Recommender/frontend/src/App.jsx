import React, { useState, useEffect, useRef } from "react";
import MovieCard from "./components/MovieCard";
import Recommendations from "./components/Recommendations";

const MIN_RATINGS = 5;

export default function App() {
  // ---------- state ----------
  const [popular, setPopular] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [ratings, setRatings] = useState({}); // { movieId: rating }
  const [recommendations, setRecommendations] = useState(null); // null | []
  const [loading, setLoading] = useState(false);

  const debounceRef = useRef(null);

  // ---------- load popular on mount ----------
  useEffect(() => {
    fetch("/api/movies/popular?n=50")
      .then((r) => r.json())
      .then(setPopular)
      .catch(console.error);
  }, []);

  // ---------- debounced search ----------
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);

    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }

    debounceRef.current = setTimeout(() => {
      fetch(`/api/movies/search?q=${encodeURIComponent(searchQuery)}`)
        .then((r) => r.json())
        .then(setSearchResults)
        .catch(console.error);
    }, 300);

    return () => clearTimeout(debounceRef.current);
  }, [searchQuery]);

  // ---------- handlers ----------
  const handleRate = (movieId, value) => {
    setRatings((prev) => {
      const next = { ...prev };
      if (value === 0) {
        delete next[movieId];
      } else {
        next[movieId] = value;
      }
      return next;
    });
  };

  const handleRecommend = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ratings, n: 20 }),
      });
      const data = await res.json();
      setRecommendations(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setRatings({});
    setRecommendations(null);
  };

  const numRated = Object.keys(ratings).length;
  const canRecommend = numRated >= MIN_RATINGS;
  const showingSearch = searchQuery.trim().length > 0;
  const displayMovies = showingSearch ? searchResults : popular;

  // ---------- render ----------
  return (
    <div className="app">
      <header className="header">
        <h1>🎬 Movie Recommender</h1>
        <p className="subtitle">
          Rate at least <strong>{MIN_RATINGS}</strong> movies, then get
          personalised recommendations powered by ALS matrix factorisation.
        </p>
      </header>

      {/* ---------- Recommendations panel ---------- */}
      {recommendations && (
        <Recommendations
          items={recommendations}
          onBack={handleReset}
        />
      )}

      {/* ---------- Rating / browsing UI ---------- */}
      {!recommendations && (
        <>
          {/* Sticky bar */}
          <div className="toolbar">
            <input
              className="search-input"
              type="text"
              placeholder="Search movies…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />

            <div className="toolbar-right">
              <span className="badge">
                {numRated} rated
              </span>
              <button
                className="recommend-btn"
                disabled={!canRecommend || loading}
                onClick={handleRecommend}
              >
                {loading
                  ? "Computing…"
                  : canRecommend
                  ? "Get Recommendations →"
                  : `Rate ${MIN_RATINGS - numRated} more`}
              </button>
            </div>
          </div>

          <h2 className="section-title">
            {showingSearch
              ? `Search results for "${searchQuery}"`
              : "Top 50 Popular Movies"}
          </h2>

          {displayMovies.length === 0 && showingSearch && (
            <p className="empty">No movies found.</p>
          )}

          <div className="grid">
            {displayMovies.map((m) => (
              <MovieCard
                key={m.movieId}
                movie={m}
                rating={ratings[m.movieId] || 0}
                onRate={(val) => handleRate(m.movieId, val)}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}