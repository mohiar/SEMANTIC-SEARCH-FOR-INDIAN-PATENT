import Head from 'next/head';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import PatentSearchForm from '../components/PatentSearchForm';
import CaptchaDisplay from '../components/CaptchaDisplay';
import ResultsDisplay from '../components/ResultsDisplay';
import styles from '../styles/Search.module.css';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export default function PatentSearchPage() {
  const [captchaImage, setCaptchaImage] = useState('');
  const [requestId, setRequestId] = useState('');
  const [results, setResults] = useState(null);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');
  const [info, setInfo] = useState('');
  const [loadingCaptcha, setLoadingCaptcha] = useState(false);
  const [searching, setSearching] = useState(false);
  const didLoadCaptcha = useRef(false);
  const canDownload = Boolean(requestId) && (results?.status === 'completed' || results?.status === 'success');

  const loadCaptcha = async () => {
    try {
      setLoadingCaptcha(true);
      setError('');

      const res = await fetch(`${API_BASE_URL}/patents/initiate`, {
        method: 'POST'
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Failed to initialize patent search');
      }

      setCaptchaImage(data.captcha_image || '');
    } catch (err) {
      setError(err.message || 'Something went wrong while loading CAPTCHA');
    } finally {
      setLoadingCaptcha(false);
    }
  };

  useEffect(() => {
    if (didLoadCaptcha.current) {
      return;
    }
    didLoadCaptcha.current = true;
    loadCaptcha();
  }, []);

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
          throw new Error(data.detail || 'Failed to fetch result status');
        }
        if (cancelled) {
          return;
        }

        setStatus(data.status || 'processing');
        if (data.status === 'success' || data.status === 'completed' || data.status === 'failed') {
          setResults(data);
          setSearching(false);
          clearInterval(interval);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || 'Unable to check patent search status');
          setSearching(false);
          clearInterval(interval);
        }
      }
    }, 2000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [requestId]);

  const onSubmit = async ({ title, captchaValue, includePapers, topK, emailId }) => {
    try {
      setError('');
      setInfo('');
      setResults(null);
      setSearching(true);
      setStatus('processing');

      if (emailId) {
        setInfo(`We will mail the results to ${emailId} once the pipeline is completed.`);
      }

      const res = await fetch(`${API_BASE_URL}/patents/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          title,
          captcha_value: captchaValue,
          include_papers: includePapers,
          top_k: Number(topK),
          email_id: emailId
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Failed to submit patent search');
      }

      setRequestId(data.request_id);
    } catch (err) {
      setSearching(false);
      setError(err.message || 'Unable to run patent search');
    }
  };

  return (
    <>
      <Head>
        <title>Patent Search | Semantic Patent Search</title>
      </Head>

      <main className={styles.page}>
        <div className={styles.headerRow}>
          <h1>Patent + Semantic Search</h1>
          <Link href="/" className={styles.backLink}>
            Back to home
          </Link>
        </div>

        {error && <p className={styles.error}>{error}</p>}
        {info && <p className={styles.info}>{info}</p>}

        <div className={styles.grid}>
          <section className={styles.panel}>
            <h2>Step 1: Solve CAPTCHA</h2>
            <CaptchaDisplay imageSrc={captchaImage} loading={loadingCaptcha} onRefresh={loadCaptcha} />
          </section>

          <section className={styles.panel}>
            <h2>Step 2: Search</h2>
            <PatentSearchForm onSubmit={onSubmit} loading={searching} />
            {status === 'processing' && (
              <p className={styles.note}>
                Pipeline running: patent scraping -&gt; scholar scraping -&gt; BM25 ranking...
              </p>
            )}
          </section>
        </div>

        <section className={styles.resultsPanel}>
          <h2>Results</h2>
          {canDownload && (
            <div className={styles.downloadRow}>
              <a
                href={`${API_BASE_URL}/search/${requestId}/download/combined`}
                className={styles.resultLink}
                target="_blank"
                rel="noreferrer"
              >
                Download Combined JSON
              </a>
              <a
                href={`${API_BASE_URL}/search/${requestId}/download/semantic`}
                className={styles.resultLink}
                target="_blank"
                rel="noreferrer"
              >
                Download Semantic JSON
              </a>
              <a
                href={`${API_BASE_URL}/search/${requestId}/download/zip`}
                className={styles.resultLink}
                target="_blank"
                rel="noreferrer"
              >
                Download ZIP
              </a>
            </div>
          )}
          <ResultsDisplay
            loading={searching}
            results={results?.results || []}
            status={results?.status}
            errorMessage={results?.error_message}
          />
        </section>
      </main>
    </>
  );
}
