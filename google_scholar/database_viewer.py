#!/usr/bin/env python3
"""
Google Scholar Database Viewer
==============================
A utility script to view, search, and export data from the
Google Scholar scraper database.
"""

import sqlite3
import csv
import json
from datetime import datetime
import argparse

class ScholarDatabaseViewer:
    def __init__(self, db_path="scholar_data.db"):
        self.db_path = db_path

    def connect_db(self):
        """Connect to the database."""
        try:
            return sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            return None

    def show_statistics(self):
        """Display database statistics."""
        conn = self.connect_db()
        if not conn:
            return

        cursor = conn.cursor()

        try:
            # Basic statistics
            cursor.execute("SELECT COUNT(*) FROM papers")
            total_papers = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT search_query) FROM papers")
            unique_queries = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL")
            papers_with_abstracts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM papers WHERE pdf_url IS NOT NULL")
            papers_with_pdfs = cursor.fetchone()[0]

            print("\n" + "="*50)
            print("DATABASE STATISTICS")
            print("="*50)
            print(f"Total papers: {total_papers}")
            print(f"Unique queries: {unique_queries}")
            print(f"Papers with abstracts: {papers_with_abstracts}")
            print(f"Papers with PDF links: {papers_with_pdfs}")

            # Recent searches
            cursor.execute("""
                SELECT query, papers_found, search_date 
                FROM search_history 
                ORDER BY search_date DESC 
                LIMIT 5
            """)
            recent_searches = cursor.fetchall()

            print("\nRecent searches:")
            for query, found, date in recent_searches:
                print(f"  - {query} ({found} papers) - {date}")

        except sqlite3.Error as e:
            print(f"Error retrieving statistics: {e}")
        finally:
            conn.close()

    def search_papers(self, search_term, field="title"):
        """Search papers by field."""
        conn = self.connect_db()
        if not conn:
            return

        cursor = conn.cursor()

        try:
            valid_fields = ["title", "authors", "abstract", "keywords", "search_query"]
            if field not in valid_fields:
                print(f"Invalid field. Valid fields: {', '.join(valid_fields)}")
                return

            query = f"""
                SELECT id, title, authors, paper_url, citations_count
                FROM papers 
                WHERE {field} LIKE ? 
                ORDER BY id DESC
            """

            cursor.execute(query, (f"%{search_term}%",))
            results = cursor.fetchall()

            print(f"\nFound {len(results)} papers matching '{search_term}' in {field}:")
            print("-" * 80)

            for i, (paper_id, title, authors, url, citations) in enumerate(results, 1):
                print(f"{i}. ID: {paper_id}")
                print(f"   Title: {title}")
                print(f"   Authors: {authors}")
                print(f"   Citations: {citations}")
                print(f"   URL: {url}")
                print("-" * 80)

        except sqlite3.Error as e:
            print(f"Error searching papers: {e}")
        finally:
            conn.close()

    def show_paper_details(self, paper_id):
        """Show detailed information for a specific paper."""
        conn = self.connect_db()
        if not conn:
            return

        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
            paper = cursor.fetchone()

            if not paper:
                print(f"Paper with ID {paper_id} not found.")
                return

            # Column names for reference
            columns = [desc[0] for desc in cursor.description]

            print(f"\nPaper Details (ID: {paper_id})")
            print("=" * 60)

            for i, value in enumerate(paper):
                if value and columns[i] not in ['id', 'scraped_at']:
                    print(f"{columns[i].replace('_', ' ').title()}: {value}")
                    print()

        except sqlite3.Error as e:
            print(f"Error retrieving paper details: {e}")
        finally:
            conn.close()

    def export_to_csv(self, filename=None, query_filter=None):
        """Export papers to CSV file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scholar_papers_{timestamp}.csv"

        conn = self.connect_db()
        if not conn:
            return

        cursor = conn.cursor()

        try:
            if query_filter:
                sql = "SELECT * FROM papers WHERE search_query = ? ORDER BY id DESC"
                cursor.execute(sql, (query_filter,))
            else:
                cursor.execute("SELECT * FROM papers ORDER BY id DESC")

            papers = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(columns)  # Header
                writer.writerows(papers)

            print(f"\nExported {len(papers)} papers to {filename}")

        except Exception as e:
            print(f"Error exporting to CSV: {e}")
        finally:
            conn.close()

    def export_to_json(self, filename=None, query_filter=None):
        """Export papers to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scholar_papers_{timestamp}.json"

        conn = self.connect_db()
        if not conn:
            return

        cursor = conn.cursor()

        try:
            if query_filter:
                sql = "SELECT * FROM papers WHERE search_query = ? ORDER BY id DESC"
                cursor.execute(sql, (query_filter,))
            else:
                cursor.execute("SELECT * FROM papers ORDER BY id DESC")

            papers = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            # Convert to list of dictionaries
            papers_dict = []
            for paper in papers:
                paper_dict = dict(zip(columns, paper))
                papers_dict.append(paper_dict)

            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(papers_dict, jsonfile, indent=2, ensure_ascii=False)

            print(f"\nExported {len(papers)} papers to {filename}")

        except Exception as e:
            print(f"Error exporting to JSON: {e}")
        finally:
            conn.close()

    def list_queries(self):
        """List all unique search queries."""
        conn = self.connect_db()
        if not conn:
            return

        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT search_query, COUNT(*) as paper_count, 
                       MIN(scraped_at) as first_scraped,
                       MAX(scraped_at) as last_scraped
                FROM papers 
                GROUP BY search_query 
                ORDER BY paper_count DESC
            """)

            queries = cursor.fetchall()

            print(f"\nAll Search Queries ({len(queries)} total):")
            print("-" * 80)

            for query, count, first, last in queries:
                print(f"Query: {query}")
                print(f"Papers: {count}")
                print(f"First scraped: {first}")
                print(f"Last scraped: {last}")
                print("-" * 80)

        except sqlite3.Error as e:
            print(f"Error listing queries: {e}")
        finally:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description="Google Scholar Database Viewer")
    parser.add_argument("--db", default="scholar_data.db", help="Database file path")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--search", help="Search papers by term")
    parser.add_argument("--field", default="title", help="Field to search in")
    parser.add_argument("--details", type=int, help="Show details for paper ID")
    parser.add_argument("--export-csv", help="Export to CSV file")
    parser.add_argument("--export-json", help="Export to JSON file")
    parser.add_argument("--query-filter", help="Filter export by search query")
    parser.add_argument("--list-queries", action="store_true", help="List all search queries")

    args = parser.parse_args()

    viewer = ScholarDatabaseViewer(args.db)

    if args.stats:
        viewer.show_statistics()
    elif args.search:
        viewer.search_papers(args.search, args.field)
    elif args.details:
        viewer.show_paper_details(args.details)
    elif args.export_csv:
        viewer.export_to_csv(args.export_csv, args.query_filter)
    elif args.export_json:
        viewer.export_to_json(args.export_json, args.query_filter)
    elif args.list_queries:
        viewer.list_queries()
    else:
        # Interactive mode
        print("Google Scholar Database Viewer")
        print("=" * 40)
        print("Available commands:")
        print("1. stats - Show database statistics")
        print("2. search <term> - Search papers")
        print("3. details <id> - Show paper details")
        print("4. export - Export data")
        print("5. queries - List all queries")
        print("6. quit - Exit")

        while True:
            try:
                cmd = input("\nEnter command: ").strip().lower()

                if cmd == "quit" or cmd == "q":
                    break
                elif cmd == "stats":
                    viewer.show_statistics()
                elif cmd.startswith("search "):
                    term = cmd[7:]
                    viewer.search_papers(term)
                elif cmd.startswith("details "):
                    try:
                        paper_id = int(cmd[8:])
                        viewer.show_paper_details(paper_id)
                    except ValueError:
                        print("Please provide a valid paper ID")
                elif cmd == "export":
                    format_choice = input("Export format (csv/json): ").strip().lower()
                    if format_choice == "csv":
                        viewer.export_to_csv()
                    elif format_choice == "json":
                        viewer.export_to_json()
                    else:
                        print("Invalid format. Choose 'csv' or 'json'")
                elif cmd == "queries":
                    viewer.list_queries()
                else:
                    print("Unknown command. Type 'quit' to exit.")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

if __name__ == "__main__":
    main()
