import pymupdf, os, glob

d = os.path.dirname(os.path.abspath(__file__))
for pdf in glob.glob(os.path.join(d, "*.pdf")):
    doc = pymupdf.open(pdf)
    out = pdf[:-4] + ".txt"
    with open(out, "w", encoding="utf-8") as f:
        for i, p in enumerate(doc):
            f.write("\n===== PAGE %d =====\n" % (i + 1))
            f.write(p.get_text())
    print(os.path.basename(out), "pages", len(doc), "chars", os.path.getsize(out))
