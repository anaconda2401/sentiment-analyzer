import os
import uuid
import pickle
import shutil
import tempfile
import logging
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

from chatsense import (
    parse_whatsapp_chat, extract_hour,
    PIIAnonymizer, TextPreprocessor, SentimentModel,
    ContextAnalyzer, ChatInsights, Visualizer, Evaluator,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Temp directory that holds per-session pickled DataFrames
SESSION_CACHE_DIR = os.path.join(tempfile.gettempdir(), 'chatsense_sessions')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs(SESSION_CACHE_DIR, exist_ok=True)

# Shared stateless modules (no mutable state between requests)
preprocessor = TextPreprocessor()
sentiment_model = SentimentModel()
context_analyzer = ContextAnalyzer(window_size=4)
insights_engine = ChatInsights()
viz = Visualizer(output_dir='static/images')
evaluator = Evaluator()

# Run evaluation once at startup so it is available for all result pages
log.info("Running model evaluation on synthetic dataset...")
eval_metrics, eval_per_class, cm_chart = evaluator.run_evaluation(sentiment_model, preprocessor)
log.info("Evaluation complete: accuracy=%.3f  weighted_f1=%.3f", eval_metrics['accuracy'], eval_metrics['weighted_f1'])


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'chat_file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['chat_file']
    if not file or file.filename == '':
        return redirect(url_for('index'))

    if not file.filename.lower().endswith('.txt'):
        return render_template('error.html', message="Please upload a WhatsApp .txt export file.")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)

        # --- Pipeline ---
        # Step 1: Parse
        df = parse_whatsapp_chat(filepath)
        if df.empty:
            return render_template('error.html', message="No valid messages found in the uploaded file. Please check the format.")

        # Step 2: PII anonymisation (fresh instance per request — no state leakage)
        anonymizer = PIIAnonymizer()
        df = anonymizer.process_dataframe(df)

        # Step 3: Preprocess
        df['clean_message'] = df['message'].apply(preprocessor.preprocess)

        # Step 4: Sentiment prediction
        predictions = df['clean_message'].apply(sentiment_model.predict)
        df['sentiment']         = predictions.apply(lambda r: r['sentiment'])
        df['confidence']        = predictions.apply(lambda r: r['confidence'])
        df['is_low_confidence'] = predictions.apply(lambda r: r['is_low_confidence'])
        df['raw_score']         = predictions.apply(lambda r: r['raw_score'])
        df['word_count']        = df['clean_message'].apply(lambda t: len(str(t).split()))

        # Step 5: Context-aware adjustment
        df = context_analyzer.apply_context(df, score_col='raw_score')

        # Step 6: Extra time-based metrics
        df['hour'] = df['time'].apply(extract_hour)
        busiest_hour = 'N/A'
        if not df['hour'].dropna().empty:
            b = int(df['hour'].mode()[0])
            busiest_hour = f"{b % 12 or 12} {'AM' if b < 12 else 'PM'}"

        # Step 6b: Parse dates and extract full range for the date picker
        df['_dt'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        date_min_raw = df['_dt'].dropna().min()
        date_max_raw = df['_dt'].dropna().max()
        date_min = date_min_raw.strftime('%Y-%m-%d') if pd.notna(date_min_raw) else ''
        date_max = date_max_raw.strftime('%Y-%m-%d') if pd.notna(date_max_raw) else ''

        # Step 6c: Persist the processed DataFrame for the filter endpoint
        session_id = uuid.uuid4().hex
        session_path = os.path.join(SESSION_CACHE_DIR, f'{session_id}.pkl')
        with open(session_path, 'wb') as fh:
            pickle.dump(df, fh)

        # Step 7: Chat-level insights
        stats = insights_engine.compute_insights(df)
        stats['busiest_hour']   = busiest_hour
        stats['avg_word_count'] = round(df['word_count'].mean(), 1)

        # Step 8: Visualisations (per-session output directory to avoid collisions)
        session_viz_dir = f'static/images/{session_id}'
        session_viz = Visualizer(output_dir=session_viz_dir)
        dist_chart        = session_viz.create_sentiment_distribution(df)
        trend_chart       = session_viz.create_trend_over_time(df)
        participant_chart = session_viz.create_participant_chart(df)
        conf_chart        = session_viz.create_confidence_histogram(df)

    except RuntimeError as e:
        log.error("Pipeline error: %s", e)
        return render_template('error.html', message=str(e))
    except Exception:
        log.exception("Unexpected error during analysis")
        return render_template('error.html', message="An unexpected error occurred while processing the file.")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return render_template(
        'results.html',
        stats=stats,
        dist_chart=dist_chart,
        trend_chart=trend_chart,
        participant_chart=participant_chart,
        conf_chart=conf_chart,
        eval_metrics=eval_metrics,
        eval_per_class=eval_per_class,
        cm_chart=cm_chart,
        sender_stats=stats['participant_stats'],
        date_min=date_min,
        date_max=date_max,
        session_id=session_id,
    )


@app.route('/filter', methods=['POST'])
def filter_by_date():
    """AJAX endpoint: re-slice the cached DataFrame by [from_date, to_date] and
    return updated stats + chart URLs as JSON."""
    data       = request.get_json(force=True)
    session_id = data.get('session_id', '')
    from_date  = data.get('from_date', '')
    to_date    = data.get('to_date', '')

    # Validate session
    session_path = os.path.join(SESSION_CACHE_DIR, f'{session_id}.pkl')
    if not session_id or not os.path.exists(session_path):
        return jsonify({'error': 'Session expired. Please re-upload the file.'}), 404

    try:
        with open(session_path, 'rb') as fh:
            df = pickle.load(fh)

        # Apply date filter
        if from_date and to_date:
            start = pd.to_datetime(from_date)
            end   = pd.to_datetime(to_date)
            mask  = (df['_dt'] >= start) & (df['_dt'] <= end)
            df    = df[mask].copy()

        if df.empty:
            return jsonify({'error': 'No messages found in the selected date range.'}), 200

        # Recompute busiest hour
        busiest_hour = 'N/A'
        if 'hour' in df.columns and not df['hour'].dropna().empty:
            b = int(df['hour'].mode()[0])
            busiest_hour = f"{b % 12 or 12} {'AM' if b < 12 else 'PM'}"

        stats = insights_engine.compute_insights(df)
        stats['busiest_hour']   = busiest_hour
        stats['avg_word_count'] = round(df['word_count'].mean(), 1) if 'word_count' in df.columns else 0

        # Re-generate charts into the session directory
        session_viz_dir = f'static/images/{session_id}'
        session_viz = Visualizer(output_dir=session_viz_dir)
        dist_chart        = session_viz.create_sentiment_distribution(df)
        trend_chart       = session_viz.create_trend_over_time(df)
        participant_chart = session_viz.create_participant_chart(df)
        conf_chart        = session_viz.create_confidence_histogram(df)

        # Build serialisable participant stats
        sender_stats = [
            {
                'sender':          s['sender'],
                'message_count':   s['message_count'],
                'avg_sentiment':   round(s['avg_sentiment'], 3),
                'avg_confidence':  round(s['avg_confidence'], 2),
                'label': (
                    'Positive' if s['avg_sentiment'] >= 0.05
                    else 'Negative' if s['avg_sentiment'] <= -0.05
                    else 'Neutral'
                ),
            }
            for s in stats.get('participant_stats', [])
        ]

        return jsonify({
            'stats':             {
                'total_messages':    stats['total_messages'],
                'participant_count': stats['participant_count'],
                'average_score':     stats['average_score'],
                'positive_pct':      stats['positive_pct'],
                'neutral_pct':       stats['neutral_pct'],
                'negative_pct':      stats['negative_pct'],
                'average_confidence':stats['average_confidence'],
                'busiest_hour':      stats['busiest_hour'],
            },
            'sender_stats':      sender_stats,
            'dist_chart':        dist_chart,
            'trend_chart':       trend_chart,
            'participant_chart': participant_chart,
            'conf_chart':        conf_chart,
        })

    except Exception as exc:
        log.exception('Filter error: %s', exc)
        return jsonify({'error': 'Server error while filtering data.'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
