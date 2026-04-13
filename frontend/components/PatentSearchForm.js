import { useState } from 'react';
import styles from '../styles/Search.module.css';

export default function PatentSearchForm({ onSubmit, loading, searchFields = [], defaultSearchField = 'Title', captchaUsed = false }) {
  const [title, setTitle] = useState('');
  const [searchField, setSearchField] = useState(defaultSearchField);
  const [captchaValue, setCaptchaValue] = useState('');
  const [emailId, setEmailId] = useState('');
  const [includePapers, setIncludePapers] = useState(false);
  const [iprLimit, setIprLimit] = useState(25);
  const [scholarLimit, setScholarLimit] = useState(10);
  const [topK, setTopK] = useState(15);

  const handleSubmit = (event) => {
    event.preventDefault();

    // Validate title
    if (!title.trim()) {
      alert("Title is required");
      return;
    }

    // Validate CAPTCHA - always required
    if (!captchaValue.trim()) {
      alert("CAPTCHA is required");
      return;
    }

    onSubmit({
      title,
      searchField,
      captchaValue: captchaValue.trim(),
      includePapers,
      iprLimit: Number(iprLimit),
      scholarLimit: Number(scholarLimit),
      topK: topK ? Number(topK) : null,
      emailId: emailId.trim() || null
    });
  };

  return (
    <form className={styles.form} onSubmit={handleSubmit}>
      <label htmlFor="patent-title">
        Patent Title / Query <span className={styles.requiredAsterisk}>*</span>
      </label>
      <input
        id="patent-title"
        type="text"
        value={title}
        required
        placeholder="Enter search query"
        onChange={(e) => setTitle(e.target.value)}
      />

      {searchFields.length > 0 && (
        <>
          <label htmlFor="search-field">Patent Search Field</label>
          <select
            id="search-field"
            value={searchField}
            onChange={(e) => setSearchField(e.target.value)}
            className={styles.select}
          >
            {searchFields.map((field) => (
              <option key={field} value={field}>
                {field}
              </option>
            ))}
          </select>
        </>
      )}

      <div className={styles.captchaBlock}>
        <label htmlFor="captcha-value">
          CAPTCHA Value <span className={styles.requiredAsterisk}>*</span>
        </label>

        <input
          id="captcha-value"
          type="text"
          value={captchaValue}
          required
          placeholder="Enter CAPTCHA text"
          onChange={(e) => setCaptchaValue(e.target.value)}
        />
      </div>

      <label htmlFor="ipr-limit">
        IPR Abstract Count
      </label>
      <input
        id="ipr-limit"
        type="number"
        min="1"
        max="200"
        value={iprLimit}
        onChange={(e) => setIprLimit(e.target.value)}
      />

      <label className={styles.checkbox}>
        <input
          type="checkbox"
          checked={includePapers}
          onChange={(e) => setIncludePapers(e.target.checked)}
        />
        Include Google Scholar papers in ranking
      </label>

      <label htmlFor="scholar-limit">Google Scholar Abstract Count (optional)</label>
      <input
        id="scholar-limit"
        type="number"
        min="1"
        max="200"
        value={scholarLimit}
        onChange={(e) => setScholarLimit(e.target.value)}
      />

      <label htmlFor="top-k">Final Top K (optional)</label>
      <input
        id="top-k"
        type="number"
        min="1"
        max="400"
        value={topK}
        placeholder="Defaults to ipr + scholar"
        onChange={(e) => setTopK(e.target.value)}
      />

      <label htmlFor="email-id">Email ID (optional)</label>
      <input
        id="email-id"
        type="email"
        value={emailId}
        placeholder="name@example.com"
        onChange={(e) => setEmailId(e.target.value)}
      />

      <button type="submit" disabled={loading || captchaUsed}>
        {loading ? 'Running pipeline...' : captchaUsed ? 'Refresh CAPTCHA to Search Again' : 'Run Combined Search'}
      </button>
    </form>
  );
}
