import Head from 'next/head';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import ResultsDisplay from '../components/ResultsDisplay';
import styles from '../styles/Search.module.css';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export default function SemanticSearchPage() {
  const [query, setQuery] = useState('');
  const [includePatents, setIncludePatents] = useState(true);
  const [includePapers, setIncludePapers] = useState(true);
  const [topK, setTopK] = useState(10);

  const [requestId, setRequestId] = useState('');
  const [searchResponse, setSearchResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');

  useEffect(() => {
    if (!requestId) {
      return;
    }

    let cancelled = false;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/search/${requestId}`);
        if (res.status === 404) {
          return;
        }
        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.detail || 'Failed to fetch search status');
        }
        if (cancelled) {
          return;
        }

        setStatus(data.status || 'processing');
        if (data.status === 'completed' || data.status === 'failed') {
          setSearchResponse(data);
          setLoading(false);
          clearInterval(interval);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || 'Unable to fetch search results');
          setLoading(false);
          clearInterval(interval);
        }
      }
    }, 2000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [requestId]);

  const submitSearch = async (event) => {
    event.preventDefault();
    try {
      setError('');
      setLoading(true);
      setSearchResponse(null);
      setStatus('processing');

      const res = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query,
          include_patents: includePatents,
          include_papers: includePapers,
          top_k: Number(topK)
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Failed to submit search');
      }

      setRequestId(data.request_id);
    } catch (err) {
      setLoading(false);
      setError(err.message || 'Search submission failed');
    }
  };

  return (
    <>
      <Head>
        <title>Semantic Search | Semantic Patent Search</title>
      </Head>

      <main className={styles.page}>
        <div className={styles.headerRow}>
          <h1>Semantic Search</h1>
          <Link href="/" className={styles.backLink}>
            Back to home
          </Link>
        </div>

        <section className={styles.panel}>
          <form className={styles.form} onSubmit={submitSearch}>
            <label htmlFor="query">Search Query</label>
            <textarea
              id="query"
              value={query}
              rows={4}
              required
              placeholder="Example: biodegradable polymer packaging for food safety"
              onChange={(e) => setQuery(e.target.value)}
            />

            <div className={styles.row}>
              <label className={styles.checkbox}>
                <input
                  type="checkbox"
                  checked={includePatents}
                  onChange={(e) => setIncludePatents(e.target.checked)}
                />
                Include patents
              </label>

              <label className={styles.checkbox}>
                <input
                  type="checkbox"
                  checked={includePapers}
                  onChange={(e) => setIncludePapers(e.target.checked)}
                />
                Include papers
              </label>
            </div>

            <label htmlFor="top-k">Top K Results</label>
            <input
              id="top-k"
              type="number"
              min="1"
              max="100"
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
            />

            <button type="submit" disabled={loading}>
              {loading ? 'Searching...' : 'Run Semantic Search'}
            </button>
          </form>

          {error && <p className={styles.error}>{error}</p>}
          {status === 'processing' && <p className={styles.note}>Search request is processing...</p>}
        </section>

        <section className={styles.resultsPanel}>
          <h2>Results</h2>
          <ResultsDisplay
            loading={loading}
            results={searchResponse?.results || []}
            status={searchResponse?.status}
            errorMessage={searchResponse?.error_message}
          />
        </section>
      </main>
    </>
  );
}
