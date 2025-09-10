import React, { useState, useEffect } from 'react';
import MarkmapComponent from './components/Markmap';
import './App.css'; // You can create an App.css for basic styling

function App() {
  const [markdown, setMarkdown] = useState('');
  const [documentId, setDocumentId] = useState('');
  const [userId, setUserId] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchMindmap = async () => {
    if (!documentId || !userId) {
      setError('Please enter both Document ID and User ID.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const response = await fetch('http://localhost:8000/mindmap', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ document_id: documentId, user_id: userId }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch mindmap');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let result = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        result += decoder.decode(value, { stream: true });
      }
      setMarkdown(result);
    } catch (err) {
      console.error('Error fetching mindmap:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>LLM Mindmap Generator</h1>
      <div className="input-container">
        <input
          type="text"
          placeholder="Document ID"
          value={documentId}
          onChange={(e) => setDocumentId(e.target.value)}
        />
        <input
          type="text"
          placeholder="User ID"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
        />
        <button onClick={fetchMindmap} disabled={loading}>
          {loading ? 'Generating...' : 'Generate Mindmap'}
        </button>
      </div>
      {error && <p className="error-message">{error}</p>}
      {markdown ? (
        <MarkmapComponent markdown={markdown} />
      ) : (
        <p>Enter document and user ID to generate a mindmap.</p>
      )}
    </div>
  );
}

export default App;
