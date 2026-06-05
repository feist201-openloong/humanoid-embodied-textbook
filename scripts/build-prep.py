#!/usr/bin/env python3
"""
Preprocess Markdown: convert $...$/$$...$$ to Katex HTML with escaped braces.
Calls Node.js Katex once per file for batch processing.
"""

import re, sys, subprocess, json, os

BASE = os.path.join(os.path.dirname(__file__), '..', 'website')

def convert_with_katex(content):
    """Pass content to Node.js which extracts math and replaces with Katex HTML."""
    script = f"""
const katex = require('katex');
let content = {json.dumps(content)};

// Process display math $$...$$
content = content.replace(/\\$\\$([\\s\\S]*?)\\$\\$/g, (_, latex) => {{
    try {{
        const html = katex.renderToString(latex, {{displayMode: true, throwOnError: false}});
        return html.replace(/{{/g, '&#123;').replace(/}}/g, '&#125;');
    }} catch(e) {{ return '[MATH ERROR]'; }}
}});

// Process inline math $...$  
content = content.replace(/(?<!\\$)\\$(?!\\$)([^\\$]*?)(?<!\\$)\\$(?!\\$)/g, (_, latex) => {{
    try {{
        const html = katex.renderToString(latex, {{displayMode: false, throwOnError: false}});
        return html.replace(/{{/g, '&#123;').replace(/}}/g, '&#125;');
    }} catch(e) {{ return '[MATH ERROR]'; }}
}});

console.log(JSON.stringify(content));
"""
    result = subprocess.run(['node', '-e', script],
                          capture_output=True, text=True, cwd=BASE,
                          timeout=30)
    return json.loads(result.stdout.strip())

def process_file(fname):
    print(f"Processing: {fname}")
    with open(fname) as f:
        content = f.read()
    new_content = convert_with_katex(content)
    if new_content != content:
        with open(fname, 'w') as f:
            f.write(new_content)
        print(f"  ✅ Converted")
    else:
        print(f"  ⚪ No math found")

if __name__ == '__main__':
    for fname in sys.argv[1:]:
        process_file(fname)
