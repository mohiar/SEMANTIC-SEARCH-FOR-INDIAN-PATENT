import styles from '../styles/Search.module.css';

export default function ResultsDisplay({ loading, results, status, errorMessage }) {
  if (loading) {
    return <p className={styles.note}>Loading results...</p>;
  }

  if (status === 'failed' || status === 'error') {
    return <p className={styles.error}>{errorMessage || 'Search failed.'}</p>;
  }

  if (!results || results.length === 0) {
    return <p className={styles.note}>No results yet.</p>;
  }

  return (
    <div className={styles.resultsList}>
      {results.map((result, index) => (
        <article className={styles.resultCard} key={`${result.title}-${index}`}>
          <h3>{result.title || 'Untitled'}</h3>
          <p>{result.abstract || 'No abstract available.'}</p>
          <div className={styles.meta}>
            <span>{result.source || 'Unknown source'}</span>
            {result.similarity_score !== undefined && result.similarity_score !== null && (
              <span>Score: {Number(result.similarity_score).toFixed(3)}</span>
            )}
          </div>
          {result.authors && <p className={styles.detail}>Authors: {result.authors}</p>}
          {result.application_number && (
            <p className={styles.detail}>Application No.: {result.application_number}</p>
          )}
          {result.url && (
            <a href={result.url} target="_blank" rel="noreferrer" className={styles.resultLink}>
              View Source
            </a>
          )}
        </article>
      ))}
    </div>
  );
}
