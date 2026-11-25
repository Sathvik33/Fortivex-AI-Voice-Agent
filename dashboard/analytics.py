"""
Dashboard module: Analytics and business insights using Plotly.
All data comes from NeonDB — no hardcoded values.
"""

import logging
from typing import Optional

import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

# Color scheme
COLORS = {
    "HOT": "#FF4B4B",
    "WARM": "#FFA500",
    "COLD": "#4B9EFF",
    "positive": "#00CC44",
    "neutral": "#888888",
    "negative": "#FF4B4B",
    "primary": "#6C63FF",
    "secondary": "#00D4AA",
    "background": "#0E1117",
}


def create_lead_distribution_chart(data: dict) -> go.Figure:
    """Create lead distribution pie chart."""
    labels = data.get("labels", [])
    values = data.get("values", [])
    
    if not labels or sum(values) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No lead data yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Lead Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        return fig
    
    color_map = {"HOT": COLORS["HOT"], "WARM": COLORS["WARM"], "COLD": COLORS["COLD"]}
    colors = [color_map.get(label, COLORS["primary"]) for label in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='label+percent+value',
        textfont=dict(size=14),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text="Lead Distribution", font=dict(size=18, color="white")),
        paper_bgcolor="rgba(14,17,23,0.8)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white")),
        margin=dict(t=50, b=20, l=20, r=20),
    )
    
    return fig


def create_sentiment_chart(data: dict) -> go.Figure:
    """Create sentiment distribution bar chart."""
    labels = data.get("labels", [])
    values = data.get("values", [])
    
    if not labels or sum(values) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No sentiment data yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Sentiment Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        return fig
    
    colors = [COLORS.get(label, COLORS["primary"]) for label in labels]
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=values,
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text="Customer Sentiment", font=dict(size=18, color="white")),
        paper_bgcolor="rgba(14,17,23,0.8)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(color="white", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(color="white", gridcolor="rgba(255,255,255,0.1)"),
        margin=dict(t=50, b=40, l=40, r=20),
    )
    
    return fig


def create_calls_timeline_chart(data: dict) -> go.Figure:
    """Create call volume over time chart."""
    dates = data.get("dates", [])
    counts = data.get("counts", [])
    
    if not dates:
        fig = go.Figure()
        fig.add_annotation(
            text="No call history yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Calls Over Time",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        return fig
    
    fig = go.Figure(data=[go.Scatter(
        x=dates,
        y=counts,
        mode='lines+markers',
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=8, color=COLORS["secondary"]),
        fill='tozeroy',
        fillcolor='rgba(108,99,255,0.1)',
        hovertemplate='Date: %{x}<br>Calls: %{y}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text="Call Volume (Last 30 Days)", font=dict(size=18, color="white")),
        paper_bgcolor="rgba(14,17,23,0.8)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(color="white", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(color="white", gridcolor="rgba(255,255,255,0.1)", title="Calls"),
        margin=dict(t=50, b=40, l=50, r=20),
    )
    
    return fig


def create_lead_score_gauge(score: float) -> go.Figure:
    """Create lead score gauge chart."""
    
    if score >= 80:
        color = COLORS["HOT"]
        category = "HOT"
    elif score >= 50:
        color = COLORS["WARM"]
        category = "WARM"
    else:
        color = COLORS["COLD"]
        category = "COLD"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "/100", 'font': {'size': 24, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': 'rgba(255,255,255,0.1)',
            'steps': [
                {'range': [0, 50], 'color': 'rgba(75,158,255,0.2)'},
                {'range': [50, 80], 'color': 'rgba(255,165,0,0.2)'},
                {'range': [80, 100], 'color': 'rgba(255,75,75,0.2)'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        },
        title={'text': f"Lead Score — {category}", 'font': {'color': color, 'size': 16}}
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(14,17,23,0.8)",
        font=dict(color="white"),
        margin=dict(t=40, b=20, l=20, r=20),
        height=200,
    )
    
    return fig


def format_kpi_cards(kpis: dict) -> str:
    """Format KPI data as HTML cards."""
    return f"""
    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:16px; padding:8px;">
        <div style="background:linear-gradient(135deg,#6C63FF,#4B9EFF); padding:20px; border-radius:12px; text-align:center;">
            <div style="font-size:32px; font-weight:bold; color:white;">{kpis.get('total_calls',0)}</div>
            <div style="color:rgba(255,255,255,0.8); margin-top:4px;">Total Calls</div>
        </div>
        <div style="background:linear-gradient(135deg,#00D4AA,#00CC44); padding:20px; border-radius:12px; text-align:center;">
            <div style="font-size:32px; font-weight:bold; color:white;">{kpis.get('total_customers',0)}</div>
            <div style="color:rgba(255,255,255,0.8); margin-top:4px;">Customers</div>
        </div>
        <div style="background:linear-gradient(135deg,#FF4B4B,#FF8C00); padding:20px; border-radius:12px; text-align:center;">
            <div style="font-size:32px; font-weight:bold; color:white;">{kpis.get('hot_leads',0)}</div>
            <div style="color:rgba(255,255,255,0.8); margin-top:4px;">🔥 HOT Leads</div>
        </div>
        <div style="background:linear-gradient(135deg,#FFA500,#FFD700); padding:20px; border-radius:12px; text-align:center;">
            <div style="font-size:32px; font-weight:bold; color:white;">{kpis.get('appointments_confirmed',0)}</div>
            <div style="color:rgba(255,255,255,0.8); margin-top:4px;">Appointments</div>
        </div>
    </div>
    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:16px; padding:8px;">
        <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); padding:16px; border-radius:12px; text-align:center;">
            <div style="font-size:24px; font-weight:bold; color:#FFA500;">{kpis.get('warm_leads',0)}</div>
            <div style="color:gray; margin-top:4px;">WARM Leads</div>
        </div>
        <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); padding:16px; border-radius:12px; text-align:center;">
            <div style="font-size:24px; font-weight:bold; color:#4B9EFF;">{kpis.get('cold_leads',0)}</div>
            <div style="color:gray; margin-top:4px;">COLD Leads</div>
        </div>
        <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); padding:16px; border-radius:12px; text-align:center;">
            <div style="font-size:24px; font-weight:bold; color:#00D4AA;">{kpis.get('appointment_conversion_rate',0)}%</div>
            <div style="color:gray; margin-top:4px;">Conversion Rate</div>
        </div>
        <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); padding:16px; border-radius:12px; text-align:center;">
            <div style="font-size:24px; font-weight:bold; color:#6C63FF;">{kpis.get('avg_lead_score',0)}</div>
            <div style="color:gray; margin-top:4px;">Avg Lead Score</div>
        </div>
    </div>
    """


def format_recommendations_html(recs: list) -> str:
    """Format follow-up recommendations as HTML."""
    if not recs:
        return "<p style='color:gray; padding:16px;'>No pending follow-ups. Great job!</p>"
    
    items = []
    for rec in recs:
        priority = rec.get("priority", "🟡")
        name = rec.get("name", "Unknown")
        company = rec.get("company", "")
        score = rec.get("lead_score", 0)
        action = rec.get("action", "")
        
        items.append(f"""
        <div style="background:rgba(255,255,255,0.05); border-left:4px solid #FFA500; 
                    padding:12px 16px; margin:8px 0; border-radius:0 8px 8px 0;">
            <div style="font-weight:bold; color:white;">{priority} {name} {f'· {company}' if company else ''}</div>
            <div style="color:#6C63FF; font-size:13px;">Score: {score}/100</div>
            <div style="color:rgba(255,255,255,0.7); margin-top:4px; font-size:13px;">{action}</div>
        </div>
        """)
    
    return "".join(items)
