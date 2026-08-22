import re
import pandas as pd


class DataCleaner:
    """
    Description: Handles all data cleaning steps for the Clinical QA pipeline.
    Process:
        1. Load raw unlabeled CSV from disk
        2. Select relevant columns
        3. Drop missing values
        4. Drop duplicate rows and questions
        5. Cast text columns to string
        6. Normalise text (HTML, URLs, non-ASCII, whitespace)
        7. Save cleaned CSV to disk
    Input:
        raw_csv_path: Path to the raw unlabeled CSV file (pubmedqa_unlabeled_raw.csv)
        output_csv_path: Path where the cleaned CSV will be saved
    Output:
        Cleaned CSV saved to output_csv_path
    """

    TARGET_COLUMNS = ['pubid', 'question', 'context', 'long_answer']

    def __init__(self, raw_csv_path: str, output_csv_path: str):
        self.raw_csv_path = raw_csv_path
        self.output_csv_path = output_csv_path

    def clean(self) -> str:
        """
        Load the raw CSV, run the full cleaning pipeline,
        save the result to output_csv_path, and return that path.
        """
        print(f"Loading raw data from {self.raw_csv_path}...")
        df = pd.read_csv(self.raw_csv_path)
        df = df[self.TARGET_COLUMNS].copy()

        # 1. Drop rows with missing values in text columns (pubid is always present)
        df.dropna(subset=['question', 'context', 'long_answer'], inplace=True)

        # 2. Drop fully duplicated rows (same question AND context)
        df.drop_duplicates(subset=['question', 'context'], inplace=True)

        # 3. Ensure text columns are strings
        df['question'] = df['question'].astype(str)
        df['context'] = df['context'].astype(str)
        df['long_answer'] = df['long_answer'].astype(str)

        # 4. Normalise text in all three text columns
        df['question'] = df['question'].apply(self._normalize_text)
        df['context'] = df['context'].apply(self._normalize_text)
        df['long_answer'] = df['long_answer'].apply(self._normalize_text)

        # 5. Deduplicate by question alone to prevent data leakage across splits
        before = len(df)
        df.drop_duplicates(subset=['question'], inplace=True)
        print(f"Dropped {before - len(df)} duplicate questions.")

        df = df.reset_index(drop=True)
        df.to_csv(self.output_csv_path, index=False)
        print(f"Cleaned data saved to {self.output_csv_path} ({len(df)} rows)")
        return self.output_csv_path

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalise a text string by:
          1. Stripping HTML tags (e.g. <b>, <sup>) — common in PubMed scraped abstracts
          2. Removing URLs
          3. Transliterating common non-ASCII medical symbols (Greek letters, degree, etc.)
          4. Dropping any remaining non-ASCII characters
          5. Collapsing all unusual line terminators (CRLF, CR, tab,
             Unicode Line/Para Separators) into a single space
          6. Collapsing multiple consecutive spaces into one
        """
        # 1. Strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # 2. Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # 3. Transliterate common non-ASCII medical symbols
        non_ascii_map = {
            '\u03b1': 'alpha', '\u03b2': 'beta',  '\u03b3': 'gamma',
            '\u03b4': 'delta', '\u03bc': 'mu',    '\u03c3': 'sigma',
            '\u00b0': ' degrees', '\u00b1': '+/-', '\u00d7': 'x',
            '\u2264': '<=',   '\u2265': '>=',   '\u2013': '-', '\u2014': '-',
        }
        for char, replacement in non_ascii_map.items():
            text = text.replace(char, replacement)
        # 4. Drop any remaining non-ASCII characters
        text = text.encode('ascii', errors='ignore').decode('ascii')
        # 5. Replace line terminators and tabs with a space
        text = re.sub(r'\r\n|\r|\t|\u2028|\u2029', ' ', text)
        # 6. Collapse multiple spaces into one
        text = re.sub(r' +', ' ', text)
        return text.strip()


if __name__ == '__main__':
    import os

    script_dir       = os.path.dirname(os.path.abspath(__file__))
    raw_csv_path     = os.path.join(script_dir, '..', '..', 'data', 'raw', 'pubmedqa_unlabeled_raw.csv')
    output_csv_path  = os.path.join(script_dir, '..', '..', 'data', 'cleaned', 'pubmedqa_unlabeled_cleaned.csv')

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    cleaner = DataCleaner(raw_csv_path=raw_csv_path, output_csv_path=output_csv_path)
    cleaner.clean()
