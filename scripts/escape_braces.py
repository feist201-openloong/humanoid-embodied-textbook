#!/usr/bin/env python3
"""Escape braces inside LaTeX math for MDX compatibility."""
import re, sys

def process_file(fname):
    with open(fname) as f:
        content = f.read()

    initial_count = content.count('&#123;')

    # Process $$...$$ display math - replace braces with HTML entities
    def fix_display(m):
        inner = m.group(1)
        inner = inner.replace('{', '&#123;').replace('}', '&#125;')
        return '$$' + inner + '$$'

    content = re.sub(r'\$\$(.*?)\$\$', fix_display, content, flags=re.DOTALL)

    # Process $...$ inline math
    def fix_inline(m):
        inner = m.group(1)
        inner = inner.replace('{', '&#123;').replace('}', '&#125;')
        return '$' + inner + '$'

    content = re.sub(r'\$(?!\$)(.*?)\$(?!\$)', fix_inline, content)

    escaped = content.count('&#123;') - initial_count
    print(f'  Escaped {escaped} braces in {fname}')

    with open(fname, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    for f in sys.argv[1:]:
        process_file(f)
