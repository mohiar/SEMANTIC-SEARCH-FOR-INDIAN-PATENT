import styles from '../styles/Search.module.css';

export default function CaptchaDisplay({ imageSrc, loading, onRefresh }) {
  return (
    <div>
      <div className={styles.captchaBox}>
        {loading && <p>Loading CAPTCHA...</p>}
        {!loading && imageSrc && <img src={imageSrc} alt="Patent CAPTCHA" className={styles.captchaImage} />}
        {!loading && !imageSrc && <p>CAPTCHA not available</p>}
      </div>
      <button type="button" onClick={onRefresh} disabled={loading} className={styles.secondaryButton}>
        Refresh CAPTCHA
      </button>
    </div>
  );
}
