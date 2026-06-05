#!/usr/bin/env python3
"""
Preprocess: add space after { inside $...$ and $$...$$ 
to prevent MDX from parsing them as JSX expressions.

MDX v3 only parses {identifier} as JSX, not {  identifier} with a space.
"""

import re
import sys

def process_file(fname):
    print(f"Processing: {fname}")
    with open(fname) as f:
        content = f.read()

    # Process $$...$$ display math
    def fix_display(m):
        inner = m.group(1)
        # Add space after every { that is not already followed by a space
        inner = re.sub(r'\{(?! )', '{ ', inner)
        return '$$' + inner + '$$'

    content = re.sub(r'\$\$(.*?)\$\$', fix_display, content, flags=re.DOTALL)

    # Process $...$ inline math
    def fix_inline(m):
        inner = m.group(1)
        inner = re.sub(r'\{(?! )', '{ ', inner)
        return '$' + inner + '$'

    content = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', fix_inline, content)

    # Also escape bare {word} patterns outside math
    # Only match standalone {word} that aren't part of HTML entities or LaTeX
    # These are rare in textbook context but may appear
    content = re.sub(r'(?<![\\&#;a-zA-Z])\{([a-zA-Z_]\w*)\}(?![a-zA-Z])',
                     r'&#123;\1&#125;', content)

    with open(fname, 'w') as f:
        f.write(content)
    print(f"  ✅ Done")

if __name__ == '__main__':
    for fname in sys.argv[1:]:
        process_file(fname)
