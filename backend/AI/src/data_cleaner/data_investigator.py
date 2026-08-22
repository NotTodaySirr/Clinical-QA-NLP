import pandas as pd
import re

df = pd.read_csv('backend/AI/data/raw/pubmedqa_unlabeled_raw.csv')

print('Information')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Dtypes:\n{df.dtypes}')

print('\nMissing Value')
print(df.isnull().sum())

print('\nDuplicates')
print(f'Duplicate rows: {df.duplicated().sum()}')
print(f'Duplicate questions: {df.duplicated(subset=["question"]).sum()}')

print('\nSample context value (raw repr)')
print(repr(df['context'].iloc[0][:300]))

print('\nFormatting Anomalies In Text Columns')
text_cols = ['question', 'context', 'long_answer']

patterns = {
    'Embedded \\n':         r'\n',
    'Embedded \\r\\n':      r'\r\n',
    'Embedded \\r':         r'\r',
    'Embedded \\t':         r'\t',
    'Unicode Line Sep':     r'\u2028',
    'Unicode Para Sep':     r'\u2029',
    'HTML Tags':            r'<[^>]+>',
    'Multiple spaces':      r'  +',
    'Non-ASCII chars':      r'[^\x00-\x7F]',
    'URLs':                 r'https?://\S+',
    'Purely numeric str':   r'^\d+$',
    'Extra leading space':  r'^ ',
    'Extra trailing space': r' $',
}

for col in text_cols:
    print(f'\n--- Column: {col} ---')
    series = df[col].fillna('').astype(str)
    checks = {
        'Embedded \\n':         lambda s: s.str.contains('\n', regex=False),
        'Embedded \\r\\n':      lambda s: s.str.contains('\r\n', regex=False),
        'Embedded \\r':         lambda s: s.str.contains('\r', regex=False),
        'Embedded \\t':         lambda s: s.str.contains('\t', regex=False),
        'Unicode Line/Para Sep':lambda s: s.apply(lambda x: '\u2028' in x or '\u2029' in x),
        'HTML Tags':            lambda s: s.str.contains('<[^>]+>', regex=True),
        'Multiple spaces':      lambda s: s.str.contains('  +', regex=True),
        'Non-ASCII chars':      lambda s: s.apply(lambda x: bool(re.search(r'[^\x00-\x7F]', x))),
        'URLs':                 lambda s: s.str.contains('https?://', regex=True),
        'Extra leading space':  lambda s: s.str.startswith(' '),
        'Extra trailing space': lambda s: s.str.endswith(' '),
    }
    for name, fn in checks.items():
        count = fn(series).sum()
        if count > 0:
            pct = count / len(df) * 100
            print(f'  [{name}]: {count} rows ({pct:.1f}%)')
    else:
        # if none printed, show a clean message
        pass

print('\nWord Count Stats')
for col in text_cols:
    wc = df[col].fillna('').astype(str).str.split().str.len()
    print(f'{col}: min={wc.min()}, mean={wc.mean():.1f}, max={wc.max()}')

print('\nUnparseable Context Rows')
bad_context = df['context'].fillna('').astype(str)
not_dict = bad_context[~bad_context.str.strip().str.startswith("{'contexts'")]
print(f'Rows where context does not look like a dict: {len(not_dict)}')
if len(not_dict) > 0:
    print(not_dict.head(3))
