import Head from 'next/head';
import Link from 'next/link';
import styles from '../styles/Home.module.css';

export default function HomePage() {
  return (
    <>
      <Head>
        <title>Semantic Patent Search</title>
        <meta
          name="description"
          content="Search Indian patents and academic papers from one interface"
        />
      </Head>

      <main className={styles.page}>
        <section className={styles.hero}>
          <h1>Semantic Patent Search</h1>
          <p>
            Unified workflow for Indian patent lookup, Google Scholar scraping, and semantic ranking.
          </p>
          <div className={styles.actions}>
            <Link href="/patent-search" className={styles.primaryBtn}>
              Combined Search
            </Link>
          </div>
        </section>

        <section className={styles.cards}>
          <article className={styles.card}>
            <h2>Combined Pipeline</h2>
            <p>
              Solve CAPTCHA, scrape patents, scrape Google Scholar, and get combined results.
            </p>
            <Link href="/patent-search" className={styles.linkBtn}>
              Open Combined Search
            </Link>
          </article>
        </section>
      </main>
    </>
  );
}
