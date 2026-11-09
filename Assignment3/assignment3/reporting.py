"""Small reportlab helpers for the reproducible assignment report."""
from html import escape
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak, HRFlowable, Image, KeepTogether, LongTable, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, TableStyle,
)


class Report:
    def __init__(self, destination):
        self.destination = Path(destination)
        fonts = Path(matplotlib.get_data_path())/"fonts"/"ttf"
        for name, file in [("DV", "DejaVuSans.ttf"), ("DV-Bold", "DejaVuSans-Bold.ttf"),
                           ("DV-Italic", "DejaVuSans-Oblique.ttf")]:
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, str(fonts/file)))
        pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-Bold", italic="DV-Italic", boldItalic="DV-Bold")
        self.width = 468
        self.story = []
        self.styles = dict(
            title=ParagraphStyle("title", fontName="DV-Bold", fontSize=22, leading=28, spaceAfter=14, textColor=colors.HexColor("#17334D")),
            subtitle=ParagraphStyle("subtitle", fontName="DV", fontSize=11, leading=15, spaceAfter=10),
            h1=ParagraphStyle("h1", fontName="DV-Bold", fontSize=15, leading=20, spaceBefore=16, spaceAfter=10, keepWithNext=True, textColor=colors.HexColor("#17334D")),
            h2=ParagraphStyle("h2", fontName="DV-Bold", fontSize=10.3, leading=14, spaceBefore=10, spaceAfter=5, keepWithNext=True),
            body=ParagraphStyle("body", fontName="DV", fontSize=9.2, leading=13.2, spaceAfter=7),
            note=ParagraphStyle("note", fontName="DV-Italic", fontSize=8, leading=11.2, spaceAfter=7, textColor=colors.HexColor("#4C5661")),
            cell=ParagraphStyle("cell", fontName="DV", fontSize=7.7, leading=10.1),
            head=ParagraphStyle("head", fontName="DV-Bold", fontSize=7.7, leading=10.1, textColor=colors.white),
        )

    def p(self, text, kind="body"):
        self.story.append(Paragraph(escape(str(text)).replace("\n", "<br/>"), self.styles[kind]))

    def h(self, title, level=1):
        if level == 1:
            self.story.append(CondPageBreak(180))
        self.p(title, "h1" if level == 1 else "h2")

    def part(self, label, question, answer):
        self.h(f"{label}  {question}", 2)
        if isinstance(answer, str):
            self.p(answer)
        else:
            for paragraph in answer:
                self.p(paragraph)

    def page(self):
        self.story.append(PageBreak())

    def table(self, frame, columns=None, labels=None, formats=None, widths=None, caption=None):
        if columns is not None:
            frame = frame.loc[:, columns]
        labels, formats = labels or {}, formats or {}
        if caption:
            self.p(caption, "note")
        data = [[Paragraph(escape(str(labels.get(c, c))).replace("\n", "<br/>"), self.styles["head"])
                 for c in frame.columns]]
        for row in frame.itertuples(index=False, name=None):
            cells = []
            for column, value in zip(frame.columns, row):
                if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
                    text = "n.a."
                elif column in formats:
                    formatter = formats[column]
                    text = formatter(value) if callable(formatter) else formatter.format(value)
                elif isinstance(value, (float, np.floating)):
                    text = f"{value:,.6g}"
                elif isinstance(value, (bool, np.bool_)):
                    text = "Yes" if value else "No"
                else:
                    text = str(value)
                cells.append(Paragraph(escape(text), self.styles["cell"]))
            data.append(cells)
        if widths is None:
            widths = [self.width/len(frame.columns)]*len(frame.columns)
        else:
            widths = np.asarray(widths, float)
            widths = widths/widths.sum()*self.width
        table = LongTable(data, colWidths=list(widths), repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24445D")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F8")]),
            ("LINEBELOW", (0, -1), (-1, -1), .4, colors.HexColor("#CBD3DA")),
        ]))
        if len(frame) <= 10:
            self.story.append(KeepTogether([table, Spacer(1, 8)]))
        else:
            self.story.extend([table, Spacer(1, 8)])

    def image(self, path, caption=None, max_height=330):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        width, height = ImageReader(str(path)).getSize()
        scale = min(self.width/width, max_height/height)
        # A chart caption must stay on the same page as the chart it describes.
        block = [Image(str(path), width=width*scale, height=height*scale, hAlign="LEFT")]
        if caption:
            block.append(Paragraph(escape(str(caption)).replace("\n", "<br/>"), self.styles["note"]))
        block.append(Spacer(1, 6))
        self.story.append(KeepTogether(block))

    def build(self):
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(self.destination), pagesize=(612, 792),
                                leftMargin=72, rightMargin=72, topMargin=50, bottomMargin=48,
                                title="Assignment 3 - Options, Option Risk, and Portfolio Construction",
                                author="Samuel Mingdao Ma")

        def furniture(canvas, document):
            canvas.saveState()
            canvas.setStrokeColor(colors.HexColor("#CBD3DA"))
            canvas.line(72, 38, 540, 38)
            canvas.setFont("DV", 7)
            canvas.setFillColor(colors.HexColor("#59636E"))
            canvas.drawString(72, 26, "Assignment 3 - Samuel Mingdao Ma")
            canvas.drawRightString(540, 26, str(document.page))
            canvas.restoreState()

        doc.build(self.story, onFirstPage=furniture, onLaterPages=furniture)
