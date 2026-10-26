import plotly.graph_objects as go
import json

# Data from the provided JSON
data = {
  "metrics": [
    {"metric": "Accuracy", "value": 65.0, "description": "Overall prediction correctness"},
    {"metric": "Precision", "value": 62.5, "description": "Correctly identified dropouts"},
    {"metric": "Recall", "value": 55.6, "description": "Dropouts that were caught"}
  ]
}

# Extract data for the chart
metrics = [item["metric"] for item in data["metrics"]]
values = [item["value"] for item in data["metrics"]]

# Brand colors in order
colors = ['#1FB8CD', '#DB4545', '#2E8B57']

# Create bar chart
fig = go.Figure()

for i, (metric, value) in enumerate(zip(metrics, values)):
    fig.add_trace(go.Bar(
        x=[metric],
        y=[value],
        name=metric,
        marker_color=colors[i],
        text=[f'{value}%'],
        textposition='outside'
    ))

# Update layout
fig.update_layout(
    title='SVM Model Performance',
    xaxis_title='Metrics',
    yaxis_title='Percentage (%)',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    yaxis=dict(range=[0, 100])
)

# Update traces for cliponaxis
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image('svm_performance_chart.png')