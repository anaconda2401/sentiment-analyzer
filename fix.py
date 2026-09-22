import os

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    # Fix the missing <nav> in results.html if it's gone
    if 'results.html' in filepath:
        if '<nav class="rail-right">' not in content:
            content = content.replace(
                '<button class="theme-toggle"',
                '<nav class="rail-right">\n        <button class="theme-toggle"'
            )
            # Find the closing </button> and add </nav> right after it
            content = content.replace(
                '</button>\n        <a href="/"',
                '</button>\n        <a href="/"'
            )
            # Actually, I'll just use a regex in python to wrap the right side properly
            import re
            content = re.sub(
                r'(<button class="theme-toggle".*?</button>\s*<a href="/" class="rail-action">.*?</a>)',
                r'<nav class="rail-right">\n        \1\n      </nav>',
                content,
                flags=re.DOTALL
            )
            # Remove duplicate nested navs just in case
            content = content.replace('<nav class="rail-right">\n        <nav class="rail-right">', '<nav class="rail-right">')
            
    content = content.replace('—', '&mdash;')
    content = content.replace('←', '&larr;')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

fix_file('templates/index.html')
fix_file('templates/results.html')
print("Fixed HTML files.")
