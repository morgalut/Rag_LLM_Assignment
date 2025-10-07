import React, { useState, useRef } from "react";

export default function AnswerUI() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const controllerRef = useRef(null);

  const API_BASE = "http://127.0.0.1:8080"; // FastAPI server

  async function handleSubmit(e) {
    e.preventDefault();
    setResult(null);
    setError("");
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(`❌ ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center p-8">
      <div className="w-full max-w-4xl bg-white shadow-md rounded-lg p-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">Tiny Demo</h1>

        {/* Input */}
        <form onSubmit={handleSubmit} className="mb-6">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="What does the Transformer architecture improve compared to RNNs?"
            rows={3}
            className="w-full p-3 border border-gray-300 rounded-md text-gray-800 focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={!query.trim() || isLoading}
            className="mt-3 px-5 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {isLoading ? "Loading..." : "Answer"}
          </button>
        </form>

        {/* Error */}
        {error && (
          <div className="bg-red-100 text-red-700 p-3 rounded-md mb-4">
            {error}
          </div>
        )}

        {/* Loading Spinner */}
        {isLoading && (
          <div className="flex justify-center items-center text-gray-500 my-4">
            <svg
              className="animate-spin h-5 w-5 mr-2 text-blue-600"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              ></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
              ></path>
            </svg>
            <span>Processing your question...</span>
          </div>
        )}

        {/* Results */}
        {result && !isLoading && (
          <div className="space-y-6">
            {/* Only show the answer text (no 'Answer' heading) */}
            <div className="bg-gray-100 text-gray-800 p-4 rounded-md whitespace-pre-wrap">
              {result.answer}
            </div>

            {/* Retrieved Context */}
            {result.retrieved_context && result.retrieved_context.length > 0 && (
              <div>
                <h2 className="text-xl font-semibold text-gray-700 mb-2">
                  Retrieved Context
                </h2>
                <div className="bg-gray-100 text-gray-700 p-4 rounded-md space-y-2">
                  {result.retrieved_context.map((ctx, idx) => (
                    <p key={idx}>- {ctx}</p>
                  ))}
                </div>
              </div>
            )}

            {/* Raw JSON */}
            <div>
              <h2 className="text-xl font-semibold text-gray-700 mb-2">
                Raw JSON
              </h2>
              <pre className="bg-gray-900 text-green-300 text-sm p-4 rounded-md overflow-x-auto">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
