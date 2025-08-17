from flask import Flask, render_template, request, send_file, Response, jsonify, redirect
import os
from datetime import datetime
import json
import queue
import threading
import pandas as pd
import numpy as np
import random
from feedback_analysis import load_and_clean_data, analyze_themes, calculate_frequencies, create_visualizations, generate_report
from openai import OpenAI

app = Flask(__name__)

# Global variables for progress tracking
progress_queue = queue.Queue()
current_analysis = {'progress': 0, 'status': 'Not started'}

def update_progress(status, progress):
    current_analysis['status'] = status
    current_analysis['progress'] = progress
    progress_queue.put({'status': status, 'progress': progress})

def process_feedback(file_path, api_key, analysis_method='clustering', gpt_model='gpt-4'):

    try:
        update_progress('Loading data...', 10)
        df = load_and_clean_data(file_path)

        update_progress('Analyzing themes...', 20)
        # Set the API key for OpenAI
        os.environ['OPENAI_API_KEY'] = api_key

        # Use the selected analysis method
        use_clustering = (analysis_method == 'clustering')
        method_name = 'clustering' if use_clustering else 'direct GPT'
        model_name = gpt_model.replace('gpt-', 'GPT-').replace('-turbo', ' Turbo')
        update_progress(f'Analyzing {len(df)} feedback entries using {method_name} approach with {model_name}...', 30)

        # Get max_clusters from global (set in analyze route)
        global max_clusters
        if 'max_clusters' in globals():
            max_clusters_val = max_clusters
        else:
            max_clusters_val = 10

        df = analyze_themes(df,
                  use_clustering=use_clustering,
                  gpt_model=gpt_model,  # For theme exploration
                  assignment_model='gpt-3.5-turbo',
                  max_clusters=max_clusters_val)

        update_progress('Calculating frequencies...', 70)
        main_theme_freq, sub_theme_freq = calculate_frequencies(df)

        update_progress('Creating visualizations...', 80)
        figures = create_visualizations(main_theme_freq, sub_theme_freq)

        update_progress('Generating report...', 90)
        output_path = 'analysis_output'
        os.makedirs(output_path, exist_ok=True)
        generate_report(df, main_theme_freq, sub_theme_freq, figures, output_path, gpt_model=gpt_model)

        update_progress('Analysis complete!', 100)
        print("DEBUG: Analysis completed successfully, set progress to 100%")
        return True

    except Exception as e:
        print(f"DEBUG ERROR: {str(e)}")
        update_progress(f'Error: {str(e)}', 0)
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check-api-key')
def check_api_key():
    # Check if OpenAI API key is set in environment
    has_api_key = bool(os.environ.get('OPENAI_API_KEY'))
    return jsonify({'has_api_key': has_api_key})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    form_api_key = request.form.get('api_key', '')
    env_api_key = os.environ.get('OPENAI_API_KEY', '')
    api_key = form_api_key if form_api_key else env_api_key
    analysis_method = request.form.get('analysis_method', 'clustering')
    gpt_model = request.form.get('gpt_model', 'gpt-4')
    # Get max_clusters from form (default 10)
    try:
        max_clusters = int(request.form.get('max_clusters', 10))
    except Exception:
        max_clusters = 10
    globals()['max_clusters'] = max_clusters
    
    if file.filename == '':
        return 'No file selected', 400
    
    if not api_key:
        return 'No API key provided (neither in form nor environment)', 400
    
    # Save the uploaded file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    upload_path = os.path.join('uploads', f'feedback_{timestamp}.xlsx')
    os.makedirs('uploads', exist_ok=True)
    file.save(upload_path)
    
    # Start processing in a background thread
    current_analysis['progress'] = 0
    current_analysis['status'] = 'Starting analysis...'
    
    thread = threading.Thread(target=process_feedback, args=(upload_path, api_key, analysis_method, gpt_model))
    thread.start()
    
    return 'Processing started', 200

@app.route('/progress')
def progress():
    def generate():
        while True:
            try:
                progress_data = progress_queue.get(timeout=1)
                if progress_data['progress'] == 100:
                    # Send completion event with redirect URL
                    redirect_data = {
                        'status': progress_data['status'], 
                        'progress': 100, 
                        'complete': True, 
                        'redirect_url': '/result'
                    }
                    yield f"data: {json.dumps(redirect_data)}\n\n"
                    break
                yield f"data: {json.dumps({'status': progress_data['status'], 'progress': progress_data['progress'], 'complete': False})}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'status': current_analysis['status'], 'progress': current_analysis['progress'], 'complete': False})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/download')
def download():
    # Find the most recent report file
    output_dir = 'analysis_output'
    files = os.listdir(output_dir)
    if not files:
        return 'No report available', 404
    
    latest_file = max([os.path.join(output_dir, f) for f in files], key=os.path.getctime)
    return send_file(latest_file, as_attachment=True)

@app.route('/dashboard')
def dashboard():
    """Render the dashboard page"""
    # Check if there's any analysis output available
    output_dir = 'analysis_output'
    if not os.path.exists(output_dir) or not os.listdir(output_dir):
        return render_template('index.html', error="No analysis has been completed yet. Please analyze data first.")
    
    return render_template('dashboard.html')
    
@app.route('/test-dashboard')
def test_dashboard():
    """Test route to directly access the dashboard"""
    return render_template('dashboard.html')

@app.route('/dashboard-data')
def dashboard_data():
    """Provide data for the dashboard"""
    # Find the most recent report file
    output_dir = 'analysis_output'
    files = [f for f in os.listdir(output_dir) if not (f.startswith('~$') or f.startswith('.'))]
    if not files:
        return jsonify({"error": "No analysis data available"}), 404

    latest_file = max([os.path.join(output_dir, f) for f in files], key=os.path.getctime)
    
    # Load the data
    try:
        if latest_file.lower().endswith('.xlsx'):
            df = pd.read_excel(latest_file)
        else:
            df = pd.read_csv(latest_file)
    except Exception as e:
        return jsonify({"error": f"Could not read analysis file: {str(e)}"}), 500
    
    # Generate dashboard data
    dashboard_data = {}
    
    # Basic stats
    dashboard_data['total_entries'] = len(df)
    dashboard_data['average_sentiment'] = float(df['Overall_Sentiment'].mean())
    
    # Sentiment distribution
    sentiment_values = df['Overall_Sentiment'].values
    dashboard_data['sentiment_distribution'] = {
        'positive': int(np.sum(sentiment_values > 0.3)),
        'neutral': int(np.sum((sentiment_values >= -0.3) & (sentiment_values <= 0.3))),
        'negative': int(np.sum(sentiment_values < -0.3))
    }
    
    # Main theme frequencies with sentiment breakdown
    theme_counts = df['Main Theme'].value_counts()
    dashboard_data['main_themes'] = []
    
    for theme in theme_counts.index:
        theme_data = df[df['Main Theme'] == theme]
        theme_sentiment = theme_data['Overall_Sentiment'].mean()
        
        # Count sentiment categories within this theme
        positive_count = int(np.sum(theme_data['Overall_Sentiment'] > 0.3))
        neutral_count = int(np.sum((theme_data['Overall_Sentiment'] >= -0.3) & (theme_data['Overall_Sentiment'] <= 0.3)))
        negative_count = int(np.sum(theme_data['Overall_Sentiment'] < -0.3))
        
        dashboard_data['main_themes'].append({
            "name": theme,
            "count": int(theme_counts[theme]),
            "avg_sentiment": float(theme_sentiment),
            "positive_count": positive_count,
            "neutral_count": neutral_count,
            "negative_count": negative_count
        })
    
    # Sub-theme frequencies
    sub_themes = []
    for main_theme in df['Main Theme'].unique():
        sub_theme_counts = df[df['Main Theme'] == main_theme]['Sub-theme'].value_counts()
        for sub_theme, count in sub_theme_counts.items():
            sub_themes.append({
                "main_theme": main_theme,
                "name": sub_theme,
                "count": int(count)
            })
    dashboard_data['sub_themes'] = sub_themes
    
    # Sample entries: one feedback per unique sub-theme, sorted by Main Theme and Sub-theme
    if len(df) > 0:
        sorted_df = df.sort_values(['Main Theme', 'Sub-theme', 'Answer'], ascending=[True, True, True])
        unique_samples = sorted_df.drop_duplicates(subset=['Main Theme', 'Sub-theme'], keep='first')
        dashboard_data['sample_entries'] = [
            {
                "main_theme": row['Main Theme'],
                "sub_theme": row['Sub-theme'],
                "text": row['Answer'],
                "sentiment": float(row['Overall_Sentiment'])
            }
            for _, row in unique_samples.iterrows()
        ]
    else:
        dashboard_data['sample_entries'] = []
    
    return jsonify(dashboard_data)

# Removed test-result route
@app.route('/result', methods=['GET', 'POST'])
def result():
    """Display the result page with action buttons after analysis completes"""
    # Check if there's any analysis output available
    output_dir = 'analysis_output'
    if not os.path.exists(output_dir) or not os.listdir(output_dir):
        return render_template('index.html', error="No analysis has been completed yet. Please analyze data first.")
    
    return render_template('result.html')
    
@app.route('/goto-results')
def goto_results():
    """Server-side redirect to results page"""
    # Keep this route as it's a useful fallback for redirect issues
    return redirect('/result')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
