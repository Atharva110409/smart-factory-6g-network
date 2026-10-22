import os

files_to_update = [
    'README.md',
    'CHANGELOG.md',
    'docs/assumptions.md',
    'reports/executive_summary.md',
    'reports/kpi_report_summary.txt',
    'reports/research_paper.md',
    'reports/diagnostics_threshold_note.md',
    'app/dashboard.py',
    'src/diagnostics.py',
    'src/kpi.py'
]

for file_path in files_to_update:
    if not os.path.exists(file_path):
        continue
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace old permutation numbers
    content = content.replace('p = 0.155', 'p = 0.1640')
    content = content.replace('p=0.155', 'p=0.1640')
    content = content.replace('0.1549', '0.1640')
    content = content.replace('FDR q = 0.533', 'FDR q = 0.5100')
    content = content.replace('FDR q = 0.5327', 'FDR q = 0.5100')
    content = content.replace('q = 0.533', 'q = 0.5100')
    content = content.replace('q = 0.5327', 'q = 0.5100')
    content = content.replace('p = 0.0381', 'p = 0.0378')
    content = content.replace('p = 0.038,', 'p = 0.0378,')
    content = content.replace('p=0.038', 'p=0.0378')
    content = content.replace('F = 4.31,', 'F = 4.3127,')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Updated: {file_path}')

print('All documentation and scripts aligned to reports/breakpoint_permutation_tests.csv!')
