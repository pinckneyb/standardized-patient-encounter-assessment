from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import json
import re


def create_assessment_pdf(assessment_json: str, output_path: str) -> str:
    """
    Create a professionally formatted PDF from medical assessment JSON.
    
    Args:
        assessment_json: JSON string containing the assessment data (or text with JSON)
        output_path: Path to save the PDF file
        
    Returns:
        Path to the generated PDF
    """
    try:
        clean_json = assessment_json.strip()
        
        if clean_json.startswith('```json'):
            clean_json = clean_json[7:]
        elif clean_json.startswith('```'):
            clean_json = clean_json[3:]
        
        if clean_json.endswith('```'):
            clean_json = clean_json[:-3]
        
        clean_json = clean_json.strip()
        
        json_match = re.search(r'\{.*\}', clean_json, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0)
        
        assessment_data = json.loads(clean_json)
    except (json.JSONDecodeError, AttributeError) as e:
        assessment_data = {
            "Error": f"Could not parse assessment JSON: {str(e)}",
            "Note": "Displaying raw text instead",
            "Overall": assessment_json[:500] + "..." if len(assessment_json) > 500 else assessment_json
        }
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1a365d'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c5282'),
        spaceAfter=12,
        spaceBefore=16,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        textColor=colors.HexColor('#1a202c'),
        spaceAfter=6,
        leading=14
    )
    
    story.append(Paragraph("STANDARDIZED PATIENT ENCOUNTER", title_style))
    story.append(Paragraph("Medical Student Assessment Report", subheading_style))
    story.append(Spacer(1, 0.3*inch))
    
    report_info = [
        ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Assessment Type:', 'Medical Student Performance Evaluation'],
        ['Evaluation Method:', 'AI-Assisted Video Analysis']
    ]
    
    info_table = Table(report_info, colWidths=[2*inch, 4.5*inch])
    info_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4a5568')),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#1a202c')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6)
    ]))
    
    story.append(info_table)
    story.append(Spacer(1, 0.4*inch))
    
    story.append(Paragraph("ASSESSMENT SCORES", heading_style))
    
    rubric_categories = [
        'History Taking',
        'Physical Examination', 
        'Communication Skills',
        'Clinical Reasoning',
        'Professionalism',
        'Patient Education and Closure'
    ]
    
    scores_data = [['Category', 'Score', 'Feedback']]
    
    # Style for feedback text (smaller, word-wrapped)
    feedback_style = ParagraphStyle(
        'FeedbackStyle',
        parent=body_style,
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#2d3748')
    )
    
    category_rows = {}
    for category in rubric_categories:
        if category in assessment_data:
            category_data = assessment_data[category]
            
            if isinstance(category_data, dict):
                score_raw = category_data.get('score', 'N/A')
                feedback = category_data.get('feedback', 'No feedback provided')
            else:
                score_raw = 'N/A'
                feedback = str(category_data) if category_data else 'No feedback provided'
            
            try:
                if score_raw == 'N/A' or score_raw is None or score_raw == '':
                    score_display = 'N/A'
                    score_num = 0
                else:
                    score_num = float(score_raw)
                    score_display = str(score_num)
                
                score_color = colors.HexColor('#38a169') if score_num >= 4 else \
                             colors.HexColor('#d69e2e') if score_num >= 3 else \
                             colors.HexColor('#e53e3e') if score_num > 0 else \
                             colors.HexColor('#718096')
            except (ValueError, TypeError):
                score_display = 'N/A'
                score_color = colors.HexColor('#718096')
            
            # Wrap feedback in Paragraph for proper word wrapping
            feedback_paragraph = Paragraph(feedback, feedback_style)
            
            row_idx = len(scores_data)
            scores_data.append([category, score_display, feedback_paragraph])
            category_rows[category] = (row_idx, score_color)
    
    scores_table = Table(scores_data, colWidths=[2*inch, 0.8*inch, 3.7*inch])
    
    table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e0')),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8)
    ]
    
    for category, (row_idx, score_color) in category_rows.items():
        table_style.append(('TEXTCOLOR', (1, row_idx), (1, row_idx), score_color))
        table_style.append(('FONTNAME', (1, row_idx), (1, row_idx), 'Helvetica-Bold'))
    
    scores_table.setStyle(TableStyle(table_style))
    story.append(scores_table)
    story.append(Spacer(1, 0.3*inch))
    
    if 'Overall' in assessment_data:
        story.append(Paragraph("OVERALL ASSESSMENT", heading_style))
        overall_text = assessment_data['Overall']
        story.append(Paragraph(overall_text, body_style))
        story.append(Spacer(1, 0.2*inch))
    
    story.append(Spacer(1, 0.3*inch))
    
    legend_data = [
        ['Score', 'Interpretation'],
        ['5', 'Exceptional - Exceeds expectations'],
        ['4', 'Proficient - Meets all expectations'],
        ['3', 'Developing - Meets most expectations'],
        ['2', 'Needs Improvement - Below expectations'],
        ['1', 'Unsatisfactory - Significant deficits']
    ]
    
    legend_table = Table(legend_data, colWidths=[0.8*inch, 5.7*inch])
    legend_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf2f7')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
    ]))
    
    story.append(Paragraph("SCORING LEGEND", subheading_style))
    story.append(legend_table)
    story.append(Spacer(1, 0.4*inch))
    
    footer_data = [
        ['Assessment Method:', 'AI-Assisted Video Analysis with GPT-5'],
        ['Generated By:', 'AI Video Analysis System'],
        ['Timestamp:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    ]
    
    footer_table = Table(footer_data, colWidths=[2*inch, 4.5*inch])
    footer_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#718096')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT')
    ]))
    
    story.append(footer_table)
    
    doc.build(story)
    print(f"✅ PDF report generated: {output_path}")
    return output_path
