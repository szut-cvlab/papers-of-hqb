import os
import fitz
from PIL import Image

BASE = r"f:\论文阅读\paperNotes"
RAW = os.path.join(BASE, "raw")
OUT = os.path.join(BASE, "readings", "assets", "2023-04-27-inssegdark")
os.makedirs(OUT, exist_ok=True)

def save_webp(png, webp, quality=80):
    im = Image.open(png).convert("RGB")
    im.save(webp, "WEBP", quality=quality)
    return os.path.getsize(webp)

# 1) 直接渲染两张矢量机制图（方法总览 Figure 3、AWD 示意 Figure 4）
fig_dir = os.path.join(RAW, "2023-04-27-inssegdark-latex", "figs")
for src, name in [("overview4.pdf", "overview"), ("awd.pdf", "awd")]:
    doc = fitz.open(os.path.join(fig_dir, src))
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
    png = os.path.join(OUT, name + ".png")
    pix.save(png)
    webp = os.path.join(OUT, name + ".webp")
    size = save_webp(png, webp, 80)
    os.remove(png)
    print(f"{name}: rect={page.rect}, size={size/1024:.1f}KB")
    doc.close()

# 2) 定位主结果表（Table 6 / tab:main），渲染整页并裁出表格区域
main_pdf = os.path.join(RAW, "2023-04-27-inssegdark.pdf")
doc = fitz.open(main_pdf)
hit = None
for i, page in enumerate(doc):
    rects = page.search_for("uncontrolled real world")
    if rects:
        hit = (i, rects[0])
        break
assert hit is not None, "main table caption not found"
pidx, cap_rect = hit
print(f"main table page index={pidx}, caption rect={cap_rect}, page rect={doc[pidx].rect}")

page = doc[pidx]
prect = page.rect
clip = fitz.Rect(0, 0, prect.width, cap_rect.y1 + 12)
pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False, clip=clip)
png = os.path.join(OUT, "table_main.png")
pix.save(png)
webp = os.path.join(OUT, "table_main.webp")
size = save_webp(png, webp, 80)
os.remove(png)
print(f"table_main: clip={clip}, size={size/1024:.1f}KB")
doc.close()
