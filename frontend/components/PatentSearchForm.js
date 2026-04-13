import { useState } from 'react';
import styles from '../styles/Search.module.css';

export default function PatentSearchForm({ onSubmit, loading }) {
  const [title, setTitle] = useState('');
  const [captchaValue, setCaptchaValue] = useState('');
  const [emailId, setEmailId] = useState('');
  const [includePapers, setIncludePapers] = useState(true);
  const [iprLimit, setIprLimit] = useState(25);
  const [scholarLimit, setScholarLimit] = useState(25);
  const [topK, setTopK] = useState(15);

  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit({
      title,
      captchaValue,
      includePapers,
      iprLimit: Number(iprLimit),
      scholarLimit: Number(scholarLimit),
      topK: topK ? Number(topK) : null,
      emailId: emailId.trim() || null
    });
  };

  return (
    <form className={styles.form} onSubmit={handleSubmit}>
      <label htmlFor="patent-title">Patent Title</label>
      <input
        id="patent-title"
        type="text"
        value={title}
        required
        placeholder="Enter patent title"
        onChange={(e) => setTitle(e.target.value)}
      />

      <label htmlFor="captcha-value">CAPTCHA Value</label>
      <input
        id="captcha-value"
        type="text"
        value={captchaValue}
        required
        placeholder="Enter CAPTCHA text"
        onChange={(e) => setCaptchaValue(e.target.value)}
      />

      <label htmlFor="ipr-limit">IPR Abstract Count</label>
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

      <label htmlFor="scholar-limit">Google Scholar Abstract Count</label>
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

      <button type="submit" disabled={loading}>
        {loading ? 'Running pipeline...' : 'Run Combined Search'}
      </button>
    </form>
  );
}
