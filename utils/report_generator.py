"""Generate downloadable PDF cost-estimation reports."""
from __future__ import annotations

from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
import matplotlib.pyplot as plt

from utils.helper import money, timestamp
from utils.visualization import cost_breakdown_chart

def create_pdf_report(inputs: dict, results: dict, recommendations: list[str]) -> bytes:
    """Create a professional PDF report and return its bytes."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], alignment=TA_CENTER,
                              textColor=colors.HexColor("#173B67"), fontSize=22, leading=27, spaceAfter=8))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#2563EB"),
                              spaceBefore=12, spaceAfter=6))
    story = [
        Paragraph("BuildSmart AI", styles["CoverTitle"]),
        Paragraph("Construction Cost Estimation & Economic Analysis", styles["Heading3"]),
        Paragraph(f"Generated: {timestamp()}", styles["Normal"]),
        Spacer(1, 8*mm),
        Paragraph("Important note", styles["Section"]),
        Paragraph("This report is an indicative estimate generated from a synthetic demonstration dataset. "
                   "It is not a bill of quantities, tender price, or professional cost certification. "
                   "Validate quantities, taxes, site conditions, and local rates with a qualified professional.", styles["BodyText"]),
        Paragraph("Project inputs", styles["Section"]),
    ]
    input_rows = [["Parameter", "Value"]]
    for key, value in inputs.items():
        input_rows.append([str(key), str(value)])
    table = Table(input_rows, colWidths=[65*mm, 105*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#173B67")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#D5DEE8")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F3F7FC")]),
        ("VALIGN", (0,0), (-1,-1), "TOP"), ("FONTSIZE", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ]))
    story.extend([table, Paragraph("Prediction and economic analysis", styles["Section"])])
    result_rows = [["Metric", "Estimate"]]
    keys = [
        ("Material Cost", "Material Cost"), ("Labour Cost", "Labour Cost"),
        ("Equipment Cost", "Equipment Cost"), ("Miscellaneous Cost", "Miscellaneous Cost"),
        ("Total Cost", "Total Cost"), ("Recommended Project Budget", "Recommended Project Budget"),
        ("Cost Per Square Foot", "Cost Per Square Foot"), ("Budget Category", "Budget Category"),
        ("Cost Efficiency Score", "Cost Efficiency Score"), ("Risk Level", "Risk Level"),
    ]
    for label, key in keys:
        value = results[key]
        if "Cost" in key and isinstance(value, (int, float)):
            value = money(value)
        elif key == "Cost Efficiency Score":
            value = f"{value}/100"
        result_rows.append([label, str(value)])
    table2 = Table(result_rows, colWidths=[75*mm, 95*mm], repeatRows=1)
    table2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#D5DEE8")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F3F7FC")]),
        ("FONTSIZE", (0,0), (-1,-1), 8), ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(table2)
    story.append(Paragraph("AI-assisted recommendations", styles["Section"]))
    for recommendation in recommendations:
        story.append(Paragraph(f"• {recommendation}", styles["BodyText"]))
        story.append(Spacer(1, 2*mm))
    story.extend([Spacer(1, 4*mm), Paragraph(f"Model: {results.get('Model Name', 'Unknown')}", styles["Italic"])])
    doc.build(story)
    return buffer.getvalue()
