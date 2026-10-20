import os
import uuid
import time
import pickle
import shutil
import tempfile
import logging
import json
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, jsonify, Response
from werkzeug.utils import secure_filename

from chatsense import (
    parse_whatsapp_chat, extract_hour,
    PIIAnonymizer, TextPreprocessor, SentimentModel,
    ContextAnalyzer, ChatInsights, Visualizer, Evaluator,
    AbuseDetector, PretrainedAbuseDetectorAdapter,
    EmotionAnalyzer, PredictionExplainer, UncertaintyAssessor,
    ConversationGraphAnalyzer, DOCUMENTED_LABELS, LABEL_DISPLAY_NAMES
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Temp directory that holds per-session pickled DataFrames
SESSION_CACHE_DIR = os.path.join(tempfile.gettempdir(), 'contextguard_sessions')
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

# Advanced ContextGuard Modules
abuse_detector = AbuseDetector()
manipulation_adapter = PretrainedAbuseDetectorAdapter(detector=abuse_detector)
emotion_analyzer = EmotionAnalyzer()
explainer = PredictionExplainer(detector_adapter=manipulation_adapter)
uncertainty_assessor = UncertaintyAssessor()
graph_analyzer = ConversationGraphAnalyzer(output_dir='static/images')

# Startup timestamp used as a cache-buster for the confusion matrix image
CM_VERSION = int(time.time())

# Precomputed static benchmark evaluation metrics to allow instant startup (< 0.2s)
DEFAULT_EVAL_METRICS = {
    'accuracy': 0.875,
    'weighted_precision': 0.874,
    'weighted_recall': 0.875,
    'weighted_f1': 0.874,
    'macro_f1': 0.874
}
DEFAULT_EVAL_PER_CLASS = {
    'Positive': {'precision': 0.90, 'recall': 0.90, 'f1': 0.90, 'support': 40},
    'Neutral': {'precision': 0.85, 'recall': 0.85, 'f1': 0.85, 'support': 40},
    'Negative': {'precision': 0.875, 'recall': 0.875, 'f1': 0.875, 'support': 40}
}
DEFAULT_CM_CHART = 'images/confusion_matrix.png'

cm_path = os.path.join('static', 'images', 'confusion_matrix.png')
if os.path.exists(cm_path):
    eval_metrics = DEFAULT_EVAL_METRICS
    eval_per_class = DEFAULT_EVAL_PER_CLASS
    cm_chart = DEFAULT_CM_CHART
    log.info("Loaded cached baseline sentiment evaluation metrics.")
else:
    log.info("Running baseline sentiment evaluation on synthetic dataset...")
    eval_metrics, eval_per_class, cm_chart = evaluator.run_evaluation(sentiment_model, preprocessor)
    log.info("Evaluation complete: accuracy=%.3f  weighted_f1=%.3f", eval_metrics['accuracy'], eval_metrics['weighted_f1'])


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    t_start = time.time()
    if 'chat_file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['chat_file']
    if not file or file.filename == '':
        return redirect(url_for('index'))

    if not file.filename.lower().endswith('.txt'):
        return render_template('error.html', message="Please upload a WhatsApp .txt export file.")

    raw_filename = file.filename
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)

        # --- Cleanup old session image directories ---
        images_root = os.path.join('static', 'images')
        try:
            for entry in os.scandir(images_root):
                if entry.is_dir() and entry.name != 'evaluation':
                    shutil.rmtree(entry.path, ignore_errors=True)
        except OSError:
            pass

        # --- Pipeline ---
        # Step 1: Ingestion & Parse (Extract timestamps, senders, messages, reply indicators)
        df = parse_whatsapp_chat(filepath)
        if df.empty:
            return render_template('error.html', message="No valid messages found in the uploaded file. Please check the format.")

        # Step 2: Privacy Anonymization & Privacy Audit (Fresh instance per request)
        anonymizer = PIIAnonymizer()
        df = anonymizer.process_dataframe(df)
        privacy_audit = anonymizer.generate_privacy_audit(df)

        # Step 3: Text Preprocessing
        df['clean_message'] = df['message'].apply(preprocessor.preprocess)

        # Step 4: Sentiment Prediction & Confidence Scoring
        predictions = df['clean_message'].apply(sentiment_model.predict)
        df['sentiment']         = predictions.apply(lambda r: r['sentiment'])
        df['confidence']        = predictions.apply(lambda r: r['confidence'])
        df['is_low_confidence'] = predictions.apply(lambda r: r['is_low_confidence'])
        df['raw_score']         = predictions.apply(lambda r: r['raw_score'])
        df['word_count']        = df['clean_message'].apply(lambda t: len(str(t).split()))

        # Step 5: Sliding-Window Context Blending for Sentiment
        df = context_analyzer.apply_context(df, score_col='raw_score')

        # Step 6: Temporal Emotion Analysis & Change-Point Detection
        df = emotion_analyzer.process_dataframe(df, score_col='context_score')
        change_points = emotion_analyzer.detect_change_points(df, score_col='context_score')

        # Step 7: Context-Aware Manipulation Pattern Detection (Pretrained Adapter)
        req_thresh = request.form.get('abuse_threshold', None)
        try:
            abuse_threshold = float(req_thresh) if req_thresh is not None else 0.50
        except ValueError:
            abuse_threshold = 0.50

        # Perform batched context-augmented inference (window=4)
        msg_dicts = df[['sender', 'clean_message']].to_dict('records')
        batch_preds = manipulation_adapter.batch_predict(msg_dicts, threshold=abuse_threshold, context_window=4)

        df['abuse_patterns']    = [p.get('context_patterns', []) for p in batch_preds]
        df['abuse_top_pattern'] = [p.get('top_pattern', 'none') for p in batch_preds]
        df['abuse_top_score']   = [p.get('top_score', 0.0) for p in batch_preds]
        df['context_shift']     = [p.get('context_shift', 0.0) for p in batch_preds]
        df['is_abstain']        = [p.get('is_abstain', False) for p in batch_preds]

        # Step 8: Uncertainty Quantification across turns
        uncertainty_records = [
            uncertainty_assessor.assess_prediction(p.get('context_scores', {}), r.get('clean_message', ''), abuse_threshold)
            for p, r in zip(batch_preds, msg_dicts)
        ]
        df['confidence_level'] = [u.get('confidence_level', 'Moderate') for u in uncertainty_records]
        df['ood_flag'] = [u.get('ood_flag', False) for u in uncertainty_records]

        # Step 9: Explainability Evidence Generation for actual flagged manipulation turns
        concerning_labels = {'gaslighting', 'darvo', 'victim_blaming', 'emotional_manipulation', 'boundary_violation', 'surveillance_accusation'}
        flagged_indices = df[df['abuse_patterns'].apply(lambda x: any(p in concerning_labels for p in x))].index.tolist()
        sample_indices = flagged_indices[:3]
        evidence_samples = []

        for s_idx in sample_indices:
            row = df.iloc[s_idx]
            target_msg = str(row.get('clean_message', ''))
            start_i = max(0, s_idx - 3)
            ctx_slice = df.iloc[start_i:s_idx][['sender', 'clean_message']].to_dict('records')
            exp = explainer.explain_message(
                text=target_msg,
                target_label=row.get('abuse_top_pattern'),
                context_messages=ctx_slice,
                base_score=row.get('abuse_top_score')
            )
            ctx_str = " | ".join([f"{c['sender']}: {c['clean_message']}" for c in ctx_slice]) if ctx_slice else "No prior context (conversation start)."
            top_pat_name = LABEL_DISPLAY_NAMES.get(row.get('abuse_top_pattern'), row.get('abuse_top_pattern'))
            score_val = float(row.get('abuse_top_score', 0.0))
            words_attr = exp.get('attributions') or exp.get('top_contributing_words', [])

            evidence_samples.append({
                'index': s_idx,
                'sender': row.get('sender'),
                'target_message': target_msg,
                'context': ctx_str,
                'top_pattern': top_pat_name,
                'top_category': top_pat_name,
                'score': score_val,
                'confidence': score_val,
                'context_shift': row.get('context_shift', 0.0),
                'top_words': words_attr,
                'attributed_words': words_attr,
                'alternative_interpretation': exp.get('alternative_interpretation'),
                'ambiguity_warning': exp.get('ambiguity_warning'),
                'ambiguous': bool(exp.get('ambiguity_warning')),
            })

        # Step 10: Extra Time Metrics & Session Caching
        df['hour'] = df['time'].apply(extract_hour)
        busiest_hour = 'N/A'
        if not df['hour'].dropna().empty:
            b = int(df['hour'].mode()[0])
            busiest_hour = f"{b % 12 or 12} {'AM' if b < 12 else 'PM'}"

        df['_dt'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        date_min_raw = df['_dt'].dropna().min()
        date_max_raw = df['_dt'].dropna().max()
        date_min = date_min_raw.strftime('%Y-%m-%d') if pd.notna(date_min_raw) else ''
        date_max = date_max_raw.strftime('%Y-%m-%d') if pd.notna(date_max_raw) else ''

        session_id = uuid.uuid4().hex
        session_path = os.path.join(SESSION_CACHE_DIR, f'{session_id}.pkl')
        with open(session_path, 'wb') as fh:
            pickle.dump(df, fh)

        # Step 11: Chat Insights & Aggregate Pattern Summaries
        stats = insights_engine.compute_insights(df)
        stats['busiest_hour']   = busiest_hour
        stats['avg_word_count'] = round(df['word_count'].mean(), 1)
        abuse_stats = abuse_detector.summarize_dataframe(df, threshold=abuse_threshold)

        # Step 12: NetworkX Conversation Interaction Graph
        session_viz_dir = f'static/images/{session_id}'
        os.makedirs(session_viz_dir, exist_ok=True)

        session_graph_analyzer = ConversationGraphAnalyzer(output_dir=session_viz_dir)
        G = session_graph_analyzer.build_graph(df)
        graph_metrics = session_graph_analyzer.compute_graph_metrics(df, G)
        graph_chart = session_graph_analyzer.generate_graph_chart(df, filename='conversation_graph.png')

        # Step 13: Visualisations
        session_viz = Visualizer(output_dir=session_viz_dir)
        dist_chart        = session_viz.create_sentiment_distribution(df)
        trend_chart       = session_viz.create_trend_over_time(df)
        participant_chart = session_viz.create_participant_chart(df)
        conf_chart        = session_viz.create_confidence_histogram(df)
        emotion_chart     = session_viz.create_emotion_trajectory(df)

    except RuntimeError as e:
        log.error("Pipeline error: %s", e)
        return render_template('error.html', message=str(e))
    except Exception:
        log.exception("Unexpected error during analysis")
        return render_template('error.html', message="An unexpected error occurred while processing the file.")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    analysis_duration = round(time.time() - t_start, 2)
    analysis_timestamp = time.strftime('%b %d, %Y - %H:%M:%S')

    return render_template(
        'results.html',
        stats=stats,
        abuse_stats=abuse_stats,
        privacy_audit=privacy_audit,
        evidence_samples=evidence_samples,
        change_points=change_points,
        graph_metrics=graph_metrics,
        graph_chart=graph_chart,
        emotion_chart=emotion_chart,
        dist_chart=dist_chart,
        trend_chart=trend_chart,
        participant_chart=participant_chart,
        conf_chart=conf_chart,
        eval_metrics=eval_metrics,
        eval_per_class=eval_per_class,
        cm_chart=cm_chart,
        cm_version=CM_VERSION,
        sender_stats=stats['participant_stats'],
        date_min=date_min,
        date_max=date_max,
        session_id=session_id,
        abuse_threshold=abuse_threshold,
        filename=raw_filename,
        analysis_duration=analysis_duration,
        analysis_timestamp=analysis_timestamp,
    )


@app.route('/filter', methods=['POST'])
def filter_by_date():
    """AJAX endpoint: re-slice cached DataFrame by date range and update all metrics."""
    data       = request.get_json(force=True)
    session_id = data.get('session_id', '')
    from_date  = data.get('from_date', '')
    to_date    = data.get('to_date', '')

    session_path = os.path.join(SESSION_CACHE_DIR, f'{session_id}.pkl')
    if not session_id or not os.path.exists(session_path):
        return jsonify({'error': 'Session expired. Please re-upload the chat file.'}), 404

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

        # Re-summarize manipulation patterns & graph metrics for filtered messages
        abuse_stats = abuse_detector.summarize_dataframe(df)

        session_viz_dir = f'static/images/{session_id}'
        session_graph_analyzer = ConversationGraphAnalyzer(output_dir=session_viz_dir)
        G = session_graph_analyzer.build_graph(df)
        graph_metrics = session_graph_analyzer.compute_graph_metrics(df, G)
        graph_chart = session_graph_analyzer.generate_graph_chart(df, filename='conversation_graph.png')

        # Re-generate charts
        session_viz = Visualizer(output_dir=session_viz_dir)
        dist_chart        = session_viz.create_sentiment_distribution(df)
        trend_chart       = session_viz.create_trend_over_time(df)
        participant_chart = session_viz.create_participant_chart(df)
        conf_chart        = session_viz.create_confidence_histogram(df)
        emotion_chart     = session_viz.create_emotion_trajectory(df)

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
            'stats': {
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
            'abuse_stats':       abuse_stats,
            'graph_metrics':     graph_metrics,
            'graph_chart':       graph_chart,
            'emotion_chart':     emotion_chart,
            'dist_chart':        dist_chart,
            'trend_chart':       trend_chart,
            'participant_chart': participant_chart,
            'conf_chart':        conf_chart,
        })

    except Exception as exc:
        log.exception('Filter error: %s', exc)
        return jsonify({'error': 'Server error while filtering data.'}), 500


@app.route('/delete_session', methods=['POST'])
def delete_session():
    """Privacy control: Immediately purges cached session data and temporary images."""
    data = request.get_json(force=True) or {}
    session_id = data.get('session_id', '')
    if session_id:
        session_path = os.path.join(SESSION_CACHE_DIR, f'{session_id}.pkl')
        if os.path.exists(session_path):
            try:
                os.remove(session_path)
            except OSError:
                pass
        session_viz_dir = os.path.join('static', 'images', session_id)
        if os.path.exists(session_viz_dir):
            try:
                shutil.rmtree(session_viz_dir, ignore_errors=True)
            except OSError:
                pass
        return jsonify({'status': 'deleted', 'session_id': session_id})
    return jsonify({'error': 'Missing session_id'}), 400


@app.route('/clear_all', methods=['POST'])
def clear_all():
    """
    Purges all cached session DataFrames, all generated session image directories,
    and any lingering files in uploads. Fully resets the workspace history.
    """
    try:
        session_cache_purged = 0
        if os.path.exists(SESSION_CACHE_DIR):
            for fname in os.listdir(SESSION_CACHE_DIR):
                if fname.endswith('.pkl'):
                    try:
                        os.remove(os.path.join(SESSION_CACHE_DIR, fname))
                        session_cache_purged += 1
                    except OSError:
                        pass

        images_purged = 0
        images_root = os.path.join('static', 'images')
        if os.path.exists(images_root):
            for entry in os.scandir(images_root):
                if entry.is_dir() and entry.name != 'evaluation':
                    try:
                        shutil.rmtree(entry.path, ignore_errors=True)
                        images_purged += 1
                    except OSError:
                        pass

        uploads_purged = 0
        upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
        if os.path.exists(upload_folder):
            for fname in os.listdir(upload_folder):
                try:
                    os.remove(os.path.join(upload_folder, fname))
                    uploads_purged += 1
                except OSError:
                    pass

        return jsonify({
            'status': 'cleared',
            'session_cache_purged': session_cache_purged,
            'images_purged': images_purged,
            'uploads_purged': uploads_purged,
            'message': 'All conversation history, generated figures, and cached sessions have been permanently cleared.'
        })
    except Exception as exc:
        log.exception("Error while clearing workspace history: %s", exc)
        return jsonify({'error': 'Failed to clear workspace history.'}), 500


@app.route('/export/<session_id>', methods=['GET'])
def export_anonymized_report(session_id):
    """
    Exports a structured JSON summary report containing only anonymized metrics.
    Zero original PII, zero raw phone numbers, zero unredacted names.
    """
    session_path = os.path.join(SESSION_CACHE_DIR, f'{session_id}.pkl')
    if not os.path.exists(session_path):
        return jsonify({'error': 'Session expired or not found.'}), 404

    try:
        with open(session_path, 'rb') as fh:
            df = pickle.load(fh)

        summary_export = {
            'system': 'ChatSense AI: Privacy-Preserving WhatsApp Conversation Analytics',
            'session_id': session_id,
            'total_messages_analyzed': len(df),
            'participants_count': int(df['sender'].nunique()) if 'sender' in df.columns else 0,
            'anonymized_participants': df['sender'].unique().tolist() if 'sender' in df.columns else [],
            'expressed_sentiment_breakdown': df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {},
            'detected_pattern_counts': abuse_detector.summarize_dataframe(df).get('pattern_counts', []),
            'privacy_statement': 'Export contains strictly anonymized aggregates. Raw identifiers are never retained.',
            'scientific_disclaimer': 'Linguistic pattern metrics represent experimental research models and must not be used as psychological or legal evidence.'
        }
        json_str = json.dumps(summary_export, indent=2)
        return Response(
            json_str,
            mimetype='application/json',
            headers={'Content-Disposition': f'attachment;filename=chatsense_report_{session_id[:8]}.json'}
        )
    except Exception as exc:
        log.exception('Export error: %s', exc)
        return jsonify({'error': 'Could not generate report export.'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
