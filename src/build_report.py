"""Render report/report.md to a self-contained report/report.html (images inlined) for PDF printing."""
import base64
import os
import re
import sys

import markdown

CSS = """
@page { size: Letter; margin: 0.6in 0.65in; }
body { font-family: 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.38; color: #111; }
h1 { font-size: 17pt; margin: 0 0 4pt; }
h2 { font-size: 13pt; border-bottom: 1px solid #999; padding-bottom: 2pt; margin-top: 14pt; }
h3 { font-size: 11pt; margin-top: 10pt; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 8.3pt; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2.5pt 4pt; text-align: left; vertical-align: top; }
th { background: #eef1f5; }
code { font-family: Consolas, monospace; font-size: 8.5pt; background: #f4f4f4; padding: 0 2px; }
pre { background: #f4f4f4; padding: 6pt; font-size: 8pt; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; }
figure { margin: 6pt 0; text-align: center; page-break-inside: avoid; }
img { max-width: 100%; max-height: 4.2in; }
figcaption { font-size: 8.5pt; color: #444; }
"""


def inline_images(html, base):
    def sub(m):
        alt, src = m.group(1), m.group(2)
        data = base64.b64encode(open(os.path.join(base, src), "rb").read()).decode()
        return f'<figure><img src="data:image/png;base64,{data}"><figcaption>{alt}</figcaption></figure>'
    return re.sub(r'<p><img alt="([^"]*)" src="([^"]+)"\s*/?></p>', sub, html)


def main():
    md_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/work/hw2d/report/report.md")
    base = os.path.dirname(md_path)
    body = markdown.markdown(open(md_path, encoding="utf-8").read(), extensions=["tables", "fenced_code"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{inline_images(body, base)}</body></html>"
    out = os.path.join(base, "report.html")
    open(out, "w", encoding="utf-8").write(html)
    print(out)


if __name__ == "__main__":
    main()
