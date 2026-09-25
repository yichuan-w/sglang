"""Build LoopChain.pptx (python-pptx). Run with any python that has python-pptx installed."""
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

INK, PAPER, PAPER2, WHITE = "14202A", "F4F6F3", "E6ECEA", "FFFFFF"
MUTED, LIGHT, LINE = "4A5866", "B9CBD1", "D5DDDF"
TEAL, AMBER, AMBERDK, OK = "1F7A8C", "E0A020", "8A5F05", "1F8A4C"
DEPTH = ["CFE6EA", "8FC3CC", "4A95A3", "1D5F6E"]
LATIN, EA, MONO = "Arial", "Microsoft YaHei", "Courier New"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
MX = 0.7  # side margin (in)


def rgb(h):
    return RGBColor.from_string(h)


def set_font(run, size, color=INK, bold=False, mono=False, italic=False):
    f = run.font
    f.size, f.bold, f.italic = Pt(size), bold, italic
    f.color.rgb = rgb(color)
    f.name = MONO if mono else LATIN
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = rpr.makeelement(qn(tag), {})
            rpr.append(el)
        el.set("typeface", EA)


def text(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, space_after=6):
    """paras: list of paragraphs; each paragraph is a list of (text, style dict) runs or a single (text, style)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        runs = para if isinstance(para, list) else [para]
        for t, st in runs:
            r = p.add_run()
            r.text = t
            set_font(r, **st)
    return tb


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = rgb(color)


def box(slide, x, y, w, h, fill=WHITE, line=LINE, radius=True, shadow=False):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    if radius:
        shp.adjustments[0] = 0.06
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(1)
    else:
        shp.line.fill.background()
    if not shadow:
        shp.shadow.inherit = False
    return shp


def header(slide, eyebrow, title, dark=False):
    text(slide, MX, 0.55, 11.9, 0.35, [(eyebrow, dict(size=13, color=AMBER if dark else TEAL, bold=True))])
    text(slide, MX, 0.9, 11.9, 0.95, [(title, dict(size=30, color=PAPER if dark else INK, bold=True))])


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def table(slide, x, y, w, rows, col_w, size=14, highlight=None, mono_cols=(), align_right_from=1, colors=None):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(0.42 * nr)).table
    for j, cw in enumerate(col_w):
        gt.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = gt.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.1)
            c.margin_top = c.margin_bottom = Inches(0.04)
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(INK if i == 0 else ("FBEFD0" if highlight == i else WHITE))
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if j >= align_right_from else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            col = PAPER if i == 0 else (colors.get((i, j), INK) if colors else INK)
            set_font(r, size - (1 if i == 0 else 0), color=col, bold=(i == 0 or highlight == i),
                     mono=(j in mono_cols and i > 0))
    return gt


# ---------------------------------------------------------------- 1 cover
s = prs.slides.add_slide(BLANK); bg(s, INK)
text(s, MX, 1.3, 11, 0.4, [("LOOPED TRANSFORMER · 无损投机解码", dict(size=14, color=AMBER, bold=True))])
text(s, MX, 1.8, 11, 1.3, [("LoopChain", dict(size=72, color=PAPER, bold=True, mono=True))])
text(s, MX, 3.2, 10.5, 1.3, [("在循环深度流水线上，每次猜一整串 token，让它们一起走完所有 loop，再无损验证。",
                              dict(size=22, color=LIGHT))])
text(s, MX, 6.3, 11, 0.4, [("Ouro-1.4B / 2.6B · Raven (R=32) · Nanbeige4.2-3B      B300 · 2026-09-24",
                            dict(size=13, color=LIGHT))])
notes(s, "一句话介绍：在 LoopSpec 的深度流水线上，把“一次猜一个 token”扩成“一次猜一串”。")

# ---------------------------------------------------------------- 2 problem
s = prs.slides.add_slide(BLANK); bg(s, PAPER)
header(s, "问题", "Loop 模型参数小，但每个 token 要把同一组层跑 R 遍")
cards = [("R = 2", "Nanbeige4.2-3B", "同一组 22 层跑两遍"),
         ("R = 4", "Ouro-1.4B / 2.6B", "整个 24/48 层栈循环 4 次"),
         ("R = 32", "Raven / Huginn", "6 层 core 循环 32 次")]
cw = (13.333 - 2 * MX - 2 * 0.35) / 3
for i, (n, t, d) in enumerate(cards):
    x = MX + i * (cw + 0.35)
    box(s, x, 2.2, cw, 2.5)
    text(s, x + 0.35, 2.45, cw - 0.7, 0.8, [(n, dict(size=36, color=TEAL, bold=True, mono=True))])
    text(s, x + 0.35, 3.35, cw - 0.7, 0.45, [(t, dict(size=18, bold=True))])
    text(s, x + 0.35, 3.85, cw - 0.7, 0.7, [(d, dict(size=15, color=MUTED))])
text(s, MX, 5.2, 11.9, 1.2, [("自回归（AR）解码时，每个 token 要串行调用 R 次循环块。参数量没变大，延迟却按 R 倍增长。",
                              dict(size=18, color=MUTED))])
notes(s, "Loop 模型的卖点是参数效率，代价是 decode 延迟成倍增加。")

# ---------------------------------------------------------------- 3 rows are free (chart)
s = prs.slides.add_slide(BLANK); bg(s, PAPER)
header(s, "关键观察", "在 B300 上，一次调用多放几行 token 几乎不花时间")
cd = CategoryChartData()
cd.categories = ["1 行", "16 行", "64 行", "128 行"]
cd.add_series("24 层 GEMM 耗时 (µs)", (747, 788, 807, 798))
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(MX), Inches(2.0), Inches(6.6), Inches(4.6), cd).chart
ch.has_legend = False
ch.has_title = True
ch.chart_title.text_frame.text = "Ouro-1.4B 24 层 GEMM 耗时（µs），CUDA graph"
set_font(ch.chart_title.text_frame.paragraphs[0].runs[0], 13, color=MUTED)
pl = ch.plots[0]
pl.has_data_labels = True
pl.data_labels.font.size = Pt(14); pl.data_labels.font.color.rgb = rgb(INK)
pl.gap_width = 80
pl.series[0].format.fill.solid(); pl.series[0].format.fill.fore_color.rgb = rgb(TEAL)
va = ch.value_axis
va.maximum_scale, va.minimum_scale = 900, 0
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = rgb(LINE)
va.tick_labels.font.size = Pt(11); va.tick_labels.font.color.rgb = rgb(MUTED)
ch.category_axis.tick_labels.font.size = Pt(13); ch.category_axis.tick_labels.font.color.rgb = rgb(MUTED)
text(s, 7.8, 2.4, 4.8, 3.8, [
    ("bs=1 解码主要卡在一连串小 kernel 的启动上，GEMV 只用到约 1/3 的显存带宽。", dict(size=17, color=MUTED)),
    ("从 1 行加到 128 行，耗时只多 7%。", dict(size=17, color=MUTED)),
    ("多出来的行等于免费的算力。", dict(size=20, bold=True))], space_after=14)
notes(s, "数据来自 gemm_rows.py：Ouro-1.4B 的 24 层 GEMM 放进 CUDA graph 计时。")

# ---------------------------------------------------------------- 4 LoopSpec cap
s = prs.slides.add_slide(BLANK); bg(s, PAPER)
header(s, "现有方法", "LoopSpec 用了这些行，但每步只放进一个新 token")
cards = [("做法", "token 走到浅层 d₁ 时，用浅层输出起草下一个 token；不同深度的 token 塞进同一次调用。"),
         ("上限 γ ≤ R/d₁", "Ouro 上限 4，实测已到 3.5–3.9；Raven 上限 16，实测 12.6–14.9。"),
         ("GPU 空转 25–40%", "每个 cycle 都要 GPU→CPU 同步，再跑 Python 调度下一步。")]
for i, (t, d) in enumerate(cards):
    x = MX + i * (cw + 0.35)
    box(s, x, 2.2, cw, 2.6)
    text(s, x + 0.35, 2.45, cw - 0.7, 0.5, [(t, dict(size=19, bold=True))])
    text(s, x + 0.35, 3.05, cw - 0.7, 1.6, [(d, dict(size=15, color=MUTED))])
text(s, MX, 5.3, 11.9, 1.0, [("WaveFront Decoding（WFD）论文 §3.2 写明了同样的上限：每 T_d 次循环调用只提交一个 token。",
                              dict(size=16, color=MUTED))])
notes(s, "γ = 每 R 次循环调用提交的 token 数。LoopSpec 和 WFD 都卡在一步一个 token。")

# ---------------------------------------------------------------- 5 statement
s = prs.slides.add_slide(BLANK); bg(s, AMBER)
text(s, MX, 1.6, 11, 0.4, [("核心想法", dict(size=14, color=INK, bold=True))])
text(s, MX, 2.1, 11.6, 2.4, [("每个起草点一次猜一整串，让这串 token 在同一次调用里一起往深处走。",
                              dict(size=38, bold=True))])
text(s, MX, 4.7, 11, 1.2, [("上限从 R/d₁ 变成 (1+τ)·R/d₁，τ 是每串平均被接受的额外 token 数。", dict(size=20))])
notes(s, "这是整个方法的一句话版本。")

# ---------------------------------------------------------------- 6 diagram
s = prs.slides.add_slide(BLANK); bg(s, PAPER)
text(s, MX, 0.5, 11.9, 0.6, [("纵轴是 token 位置，横轴是每一次循环调用（R=4，d₁=1，假设全部接受）", dict(size=24, bold=True))])


def schedule(mode, k=3, P=20, R=4):
    if mode == "ar":
        return [(i, i * R, False) for i in range(P)]
    if mode == "ls":
        return [(i, i, False) for i in range(P)]
    out, pos, g = [], 0, 0
    while pos < P:
        for j in range(k + 1):
            if pos >= P:
                break
            out.append((pos, g, j > 0)); pos += 1
        g += 1
    return out


T, P, R = 8, 20, 4
cellw, cellh = 0.44, 0.215
panel_w = T * cellw
gapx = (13.333 - 2 * MX - 3 * panel_w) / 2
for pi, (mode, name) in enumerate((("ar", "AR 自回归"), ("ls", "LoopSpec"), ("lc", "LoopChain（k=3）"))):
    x0 = MX + pi * (panel_w + gapx)
    y0 = 2.05
    toks = schedule(mode)
    committed = sum(1 for pos, st, _ in toks if st + R - 1 < T)
    col = AMBERDK if mode == "lc" else INK
    text(s, x0, 1.25, panel_w, 0.4, [(name, dict(size=17, bold=True, color=col))])
    text(s, x0, 1.62, panel_w, 0.35, [(f"8 步内提交 {committed} 个 token", dict(size=13, color=MUTED))])
    frame = box(s, x0, y0, panel_w, P * cellh, fill=WHITE, line=LINE, radius=False)
    for pos, st, drafted in toks:
        for d in range(1, R + 1):
            t = st + d - 1
            if t >= T:
                break
            c = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x0 + t * cellw + 0.02), Inches(y0 + pos * cellh + 0.02),
                                   Inches(cellw - 0.04), Inches(cellh - 0.04))
            c.fill.solid(); c.fill.fore_color.rgb = rgb(DEPTH[d - 1]); c.shadow.inherit = False
            if drafted and d == 1:
                c.line.color.rgb = rgb(AMBER); c.line.width = Pt(2)
            else:
                c.line.fill.background()
            if d == R:
                dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x0 + t * cellw + cellw / 2 - 0.05),
                                         Inches(y0 + pos * cellh + cellh / 2 - 0.05), Inches(0.1), Inches(0.1))
                dot.fill.solid(); dot.fill.fore_color.rgb = rgb(OK); dot.line.fill.background(); dot.shadow.inherit = False
legend = [(DEPTH[0], "深度 1"), (DEPTH[1], "深度 2"), (DEPTH[2], "深度 3"), (DEPTH[3], "深度 4（最终）")]
lx = MX
for c, t in legend:
    sw = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(lx), Inches(6.62), Inches(0.22), Inches(0.22))
    sw.fill.solid(); sw.fill.fore_color.rgb = rgb(c); sw.line.fill.background(); sw.shadow.inherit = False
    text(s, lx + 0.3, 6.6, 1.6, 0.3, [(t, dict(size=12, color=MUTED))])
    lx += 1.75 if "最终" not in t else 2.2
sw = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(lx), Inches(6.62), Inches(0.22), Inches(0.22))
sw.fill.solid(); sw.fill.fore_color.rgb = rgb(DEPTH[0]); sw.line.color.rgb = rgb(AMBER); sw.line.width = Pt(2); sw.shadow.inherit = False
text(s, lx + 0.3, 6.6, 1.8, 0.3, [("起草的 token", dict(size=12, color=MUTED))])
lx += 1.95
dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(lx + 0.04), Inches(6.66), Inches(0.14), Inches(0.14))
dot.fill.solid(); dot.fill.fore_color.rgb = rgb(OK); dot.line.fill.background(); dot.shadow.inherit = False
text(s, lx + 0.3, 6.6, 2.0, 0.3, [("验证通过、提交", dict(size=12, color=MUTED))])
notes(s, "AR 每个 token 串行 4 步；LoopSpec 每步提交 1 个；LoopChain 同一时刻起草 4 个，一起往深处走，每步提交 4 个。只画了前 20 个位置，所以 LoopChain 在 20 处截断。")

# ---------------------------------------------------------------- 7 components
s = prs.slides.add_slide(BLANK); bg(s, PAPER)
header(s, "方法", "四个部件")
comps = [("1", "起草一整串", "每产生一个新 token，drafter 立刻再猜 k 个：n-gram、浅层 hidden 上训练的小 drafter，或两者合成一棵小树。"),
         ("2", "同调用祖先可见", "一次调用里所有行先写 KV 再算 attention；后面的 token 能看到前面 token 刚写入的 KV。"),
         ("3", "浅层早检查", "token 到深度 d₁ 时检查下一个猜测，错了当场截断重猜；浅层不可靠时按置信度关掉。"),
         ("4", "最终深度无损验证", "Greedy 只收最终 argmax；Sampling 按 min(1, p/q) 接受、残差重采样，全在 CUDA graph 内。")]
gw = (13.333 - 2 * MX - 0.35) / 2
for i, (n, t, d) in enumerate(comps):
    x = MX + (i % 2) * (gw + 0.35); y = 2.1 + (i // 2) * 2.25
    box(s, x, y, gw, 2.0)
    text(s, x + 0.3, y + 0.3, 0.6, 0.8, [(n, dict(size=32, color=AMBER, bold=True, mono=True))])
    text(s, x + 1.0, y + 0.3, gw - 1.3, 0.45, [(t, dict(size=18, bold=True))])
    text(s, x + 1.0, y + 0.8, gw - 1.3, 1.1, [(d, dict(size=14, color=MUTED))])
notes(s, "第 2 点是系统上最关键、也是 LoopSpec 现有 SGLang 代码缺的部分。")

# ---------------------------------------------------------------- 8 why beat DtV
s = prs.slides.add_slide(BLANK); bg(s, PAPER)
header(s, "对比", "为什么能赢 DFlash / EAGLE 这类先起草后验证")
text(s, MX, 2.1, 4.6, 3.8, [
    ("先起草后验证（DtV）每轮验证都要串行走完 R 层 loop，drafter 再强也只能和 LoopSpec 打平。", dict(size=16, color=MUTED)),
    ("LoopSpec 靠流水线把验证藏起来，但一步只进一个 token。", dict(size=16, color=MUTED)),
    ("两者正交，合起来是乘法。", dict(size=18, bold=True))], space_after=14)
table(s, 5.6, 2.1, 7.0, [
    ["Ouro-1.4B，γ（AR=1）", "GSM8K", "HE+", "MATH", "MBPP+"],
    ["LoopSpec(1,2)", "3.36", "3.79", "3.55", "3.30"],
    ["DtV + drafter（接受率 0.8）", "3.36", "3.53", "4.18", "3.12"],
    ["LoopChain + 同一 drafter", "7.38", "11.27", "9.77", "6.93"]],
    [3.0, 1.0, 1.0, 1.0, 1.0], size=14, highlight=3, mono_cols=(1, 2, 3, 4))
text(s, 5.6, 4.3, 7.0, 0.8, [("来源：用真实逐深度预测回放的模拟器，与实测 γ 误差约 3%；drafter 开销未计入。",
                              dict(size=12, color=MUTED))])
notes(s, "γ = 每 R 次循环调用提交的 token 数。")

# ---------------------------------------------------------------- 9 results
s = prs.slides.add_slide(BLANK); bg(s, PAPER)
header(s, "实测", "比 LoopSpec 再快 1.1–2.8 倍，且无损")
table(s, MX, 2.1, 11.9, [
    ["模型", "LoopSpec", "LoopChain", "比 LoopSpec"],
    ["Ouro-1.4B（每任务 30 条）", "3.2–3.3×", "5.2–6.2×", "1.6–1.9×"],
    ["Ouro-2.6B（每任务 10 条）", "3.0–3.6×", "5.3–10.1×", "1.6–2.8×"],
    ["Raven-Llama-3.2，R=32（每任务 6 条）", "4.4–4.8×", "4.9–6.3×", "1.1–1.3×"],
    ["Ouro-1.4B，sampling T=1", "2.2–3.2×", "2.7–5.7×", "1.1–1.8×"]],
    [5.3, 2.2, 2.2, 2.2], size=17, mono_cols=(1, 2, 3),
    colors={(i, 2): AMBERDK for i in range(1, 5)})
text(s, MX, 5.1, 11.9, 1.2, [("B300 单卡，bs=1，256 tokens，四个任务（GSM8K / HumanEval+ / MATH-500 / MBPP+）的区间；"
                              "所有方法在同一个自写 CUDA-graph 引擎里跑，倍数相对同引擎 AR。引擎里的 LoopSpec 与原版 SGLang 实现速度相当（510 vs 461 tok/s）。",
                              dict(size=13, color=MUTED))])
notes(s, "最好的配置是 n-gram 与训练的小 drafter 合成的小树（both+d2）。")

# ---------------------------------------------------------------- 10 lossless
s = prs.slides.add_slide(BLANK); bg(s, PAPER2)
header(s, "正确性", "输出和 AR 一致，只在 BF16 并列处分叉")
stats = [("80.0% → 78.5%", "Ouro-1.4B GSM8K test", "n=200，误差内"),
         ("85.0% → 86.5%", "Ouro-2.6B GSM8K test", "n=200，误差内"),
         ("0 或 0.125", "分叉点的 top1−top2 logit 差", "全体位置中位数 6.25")]
for i, (a, b, c) in enumerate(stats):
    x = MX + i * (cw + 0.35)
    box(s, x, 2.2, cw, 2.4)
    text(s, x + 0.35, 2.45, cw - 0.7, 0.7, [(a, dict(size=26, color=TEAL, bold=True, mono=True))])
    text(s, x + 0.35, 3.3, cw - 0.7, 0.5, [(b, dict(size=16, bold=True))])
    text(s, x + 0.35, 3.8, cw - 0.7, 0.6, [(c, dict(size=14, color=MUTED))])
text(s, MX, 5.1, 11.9, 1.2, [("LoopSpec 本身也在同样的位置分叉。Sampling 模式下，300 条样本的分布差异落在 AR 自己两次运行的噪声范围内。",
                              dict(size=16, color=MUTED))])
notes(s, "要做到逐 token 完全一致，需要 FP32 logits 或 batch-invariant kernel。")

# ---------------------------------------------------------------- 11 serving low B
s = prs.slides.add_slide(BLANK); bg(s, PAPER)
header(s, "服务场景", "低并发时 LoopChain 优势最大")
table(s, MX, 2.1, 6.4, [
    ["并发请求数", "LoopSpec", "LoopChain"],
    ["1", "3.33×", "7.82×"], ["2", "3.04×", "5.21×"], ["4", "2.95×", "4.39×"],
    ["8", "2.64×", "3.07×"], ["16", "2.31×", "2.30×"], ["32", "1.79×", "1.28×"]],
    [2.2, 2.1, 2.1], size=16, mono_cols=(0, 1, 2), colors={(i, 2): AMBERDK for i in range(1, 7)})
text(s, 7.6, 2.2, 5.0, 3.8, [
    ("GSM8K，Ouro-1.4B：每个请求相对同并发 AR 的延迟加速。", dict(size=16, color=MUTED)),
    ("并发 ≥16 时空闲行被占满，LoopChain 不再占优，回退到 LoopSpec。", dict(size=16, color=MUTED)),
    ("策略：按负载切换。", dict(size=18, bold=True))], space_after=14)
notes(s, "HumanEval+ 趋势相同：1 并发 6.0×，4 并发 5.2×，8 并发 3.7×。")

# ---------------------------------------------------------------- 12 saturation (chart)
s = prs.slides.add_slide(BLANK); bg(s, PAPER)
header(s, "打满时", "并发到 150–190，LoopSpec 的加速消失")
cd = CategoryChartData()
cd.categories = ["32", "64", "128", "192"]
cd.add_series("AR", (2967, 4695, 6678, 7718))
cd.add_series("LoopSpec", (5123, 6448, 7209, 7350))
ch = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(MX), Inches(2.0), Inches(7.2), Inches(4.7), cd).chart
ch.has_title = True
ch.chart_title.text_frame.text = "GSM8K 总吞吐（tok/s）vs 并发请求数，Ouro-1.4B"
set_font(ch.chart_title.text_frame.paragraphs[0].runs[0], 13, color=MUTED)
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
ch.legend.font.size = Pt(13)
for ser, col in zip(ch.plots[0].series, (MUTED, TEAL)):
    ser.format.line.color.rgb = rgb(col); ser.format.line.width = Pt(3)
    ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = rgb(col)
    ser.marker.format.line.color.rgb = rgb(col)
ch.plots[0].has_data_labels = True
ch.plots[0].data_labels.font.size = Pt(11); ch.plots[0].data_labels.font.color.rgb = rgb(INK)
va = ch.value_axis; va.minimum_scale, va.maximum_scale = 0, 9000
va.major_gridlines.format.line.color.rgb = rgb(LINE)
va.tick_labels.font.size = Pt(11); va.tick_labels.font.color.rgb = rgb(MUTED)
ch.category_axis.tick_labels.font.size = Pt(13)
text(s, 8.3, 2.2, 4.3, 4.2, [
    ("延迟加速：32 并发 1.90×，64 并发 1.44×，128 并发 1.13×，192 并发 0.99×。", dict(size=15, color=MUTED)),
    ("LoopSpec 每个请求要约 3.4 行；总行数到 400–650 后 GEMM 变成算力受限，多出的行不再免费。", dict(size=15, color=MUTED)),
    ("Loop 模型的 KV 是每层深度各一份（R 倍），batch 上限也更早被显存卡住。", dict(size=15, color=MUTED)),
    ("B ≳ 150 时用普通 AR 更好。", dict(size=17, bold=True))], space_after=12)
notes(s, "HumanEval+ 同样：32 并发 2.05×，64 并发 1.48×，128 并发 1.14×，192 并发 1.01×；AR 在 256 并发时 9919 tok/s，还在涨。")

# ---------------------------------------------------------------- 13 limits
s = prs.slides.add_slide(BLANK); bg(s, PAPER)
header(s, "边界", "什么时候效果不好")
lims = [("Raven：core 太小", "6 层 1B 规模的 core，多加行不再完全免费；收益 1.1–1.3×，短 chain（k=2–4）更好。"),
        ("Nanbeige：浅层 readout 太差", "loop 1 与最终只有 45% 一致；靠自身 loop 起草封顶约 1.3×，要靠 EAGLE3 / DSpark 并关掉早检查。"),
        ("高并发", "8–16 以上 LoopChain 不占优；150–190 以上连 LoopSpec 也不如普通 AR。")]
for i, (a, b) in enumerate(lims):
    y = 2.1 + i * 1.45
    box(s, MX, y, 11.9, 1.2)
    text(s, MX + 0.35, y + 0.2, 3.9, 0.8, [(a, dict(size=18, bold=True))], anchor=MSO_ANCHOR.MIDDLE)
    text(s, MX + 4.4, y + 0.2, 7.2, 0.8, [(b, dict(size=15, color=MUTED))], anchor=MSO_ANCHOR.MIDDLE)
notes(s, "Nanbeige 的 intermediate loop 没有训练 LM loss（loop_loss_weights 为空）。")

# ---------------------------------------------------------------- 14 next
s = prs.slides.add_slide(BLANK); bg(s, INK)
header(s, "下一步", "离一篇论文还差这四件事", dark=True)
items = ["在 SGLang 里实现：子 token 的 KV 时间线要包含父 token 同一次调用里新分配的 slot",
         "完整测试集，加上 Huginn-0125 本体和 Ouro Thinking 长 CoT",
         "更强的 drafter（DFlash block drafter），以及能控制行数的早检查策略",
         "逐 token 完全一致：FP32 logits 或 batch-invariant kernel"]
for i, t in enumerate(items):
    y = 2.1 + i * 0.95
    text(s, MX, y, 0.6, 0.6, [(str(i + 1), dict(size=24, color=AMBER, bold=True, mono=True))])
    text(s, MX + 0.7, y + 0.04, 11.0, 0.8, [(t, dict(size=18, color=PAPER))])
text(s, MX, 6.3, 11.9, 0.5, [("代码与完整报告：loopspec/research/REPORT.md（本地分支 research/multitoken-chains）",
                              dict(size=13, color=LIGHT))])
notes(s, "SGLang 改动点：graph_ops.update_request_mapping 和 BranchManager.fork，并允许每个 cycle 多个 lane 到达最终深度。")

out = "/ssd1/k3/yichuan/sglang/loop-spec-research/LoopChain.pptx"
prs.save(out)
print("saved", out, len(prs.slides), "slides")
