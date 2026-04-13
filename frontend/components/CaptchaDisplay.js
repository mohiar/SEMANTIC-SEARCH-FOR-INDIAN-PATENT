import styles from '../styles/Search.module.css';

export default function CaptchaDisplay({ imageSrc, loading, onRefresh, isSearching, captchaUsed }) {
  return (
    <div>
      <div className={captchaUsed && !loading ? styles.captchaBoxWithOverlay : styles.captchaBox}>
        {loading && <p>Loading CAPTCHA...</p>}
        {!loading && imageSrc && (
          <>
            <img src={imageSrc} alt="Patent CAPTCHA" className={styles.captchaImage} />
            {captchaUsed && !loading && (
              <div className={styles.captchaOverlay}>
                <button 
                  type="button" 
                  onClick={onRefresh} 
                  disabled={loading} 
                  className={styles.refreshButtonOverlay}
                >
                  🔄 Refresh CAPTCHA
                </button>
              </div>
            )}
          </>
        )}
        {!loading && !imageSrc && <p>CAPTCHA not available</p>}
      </div>
      {!captchaUsed && (
        <button 
          type="button" 
          onClick={onRefresh} 
          disabled={loading} 
          className={styles.secondaryButton}
        >
          Refresh CAPTCHA
        </button>
      )}
      {captchaUsed && !loading && (
        <p style={{ marginTop: '8px', fontSize: '0.9em', color: '#d9534f' }}>
          ⚠️ Refresh CAPTCHA above to run another search
        </p>
      )}
    </div>
  );
}
