from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from io import BytesIO
import json
import threading

import os
import gdown

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rfdetr import RFDETRMedium

ROOT = Path(__file__).resolve().parent


CHECKPOINT_PATH = ROOT / "checkpoint_best_total.pth"

# Google Drive file ID
FILE_ID = "1JKVnGlvQCmkVzvx4Ep8rC6Z-ThnXSM8P"

def download_checkpoint():
    if CHECKPOINT_PATH.is_file() and CHECKPOINT_PATH.stat().st_size > 0:
        return CHECKPOINT_PATH

    url = f"https://drive.google.com/uc?id={FILE_ID}"
    temp_path = CHECKPOINT_PATH.with_suffix(".download")

    try:
        with st.spinner("Downloading model checkpoint..."):
            gdown.download(
                url,
                str(temp_path),
                quiet=True,
                fuzzy=True,
            )

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError("Checkpoint download failed.")

        # Basic check against downloading an HTML error page
        with open(temp_path, "rb") as f:
            header = f.read(256).lower()
        if b"<html" in header or b"<!doctype html" in header:
            raise RuntimeError("Google Drive returned HTML instead of model weights.")

        os.replace(temp_path, CHECKPOINT_PATH)
        return CHECKPOINT_PATH

    finally:
        if temp_path.exists():
            temp_path.unlink()



NAMES = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
COLORS = ['#15803d', '#c98518', '#089a91', '#7354c9', '#168cbb', '#d05e8b']
TZ = ZoneInfo('Asia/Riyadh')
st.set_page_config(page_title='Waste to Value', page_icon='🌱', layout='wide')
st.markdown('''<style>
.stApp {background:#f7f7ef;color:#173c2e;}
.block-container {max-width:1180px;padding-top:2.5rem;}
h1,h2,h3 {color:#173c2e;}
.hero {background:#173c2e;color:#fff;padding:34px 40px;border-radius:24px;margin-bottom:28px;}
.hero h1 {color:#fff;margin:0;font-size:42px;}
.hero p {color:#d1e4d2;margin:10px 0 0;font-size:17px;}
.eyebrow {font-size:12px;letter-spacing:3px;color:#b4d6a1;margin-bottom:12px;}
[data-testid="stMetric"] {background:#fff;padding:20px;border:1px solid #dde5d5;border-radius:18px;}
[data-testid="stFileUploader"] {background:#fff;border-radius:18px;padding:12px;}
.stButton>button {border-radius:12px;min-height:48px;}
</style>''', unsafe_allow_html=True)
st.markdown('''<div class="hero"><div class="eyebrow">SMART WASTE ASSESSMENT</div>
<h1>🌱 Waste to Value</h1><p>From a waste image to a clear next step.</p></div>''', unsafe_allow_html=True)

@st.cache_resource
def load_model(path, stamp):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings = json.loads((ROOT / 'training_config.json').read_text())
    if settings.get('class_names') != NAMES:
        raise ValueError('Training class names do not match the waste dataset.')
    architecture = settings['model_config']
    model = RFDETRMedium(
        pretrain_weights=path, device=device, num_classes=6,
        resolution=architecture['resolution'],
        positional_encoding_size=architecture['positional_encoding_size'],
    )
    return model, threading.Lock()

def draw_boxes(im, boxes, classes, scores):
    result = im.copy()
    draw = ImageDraw.Draw(result)
    size = max(12, round(min(im.size) / 40))
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        font = ImageFont.load_default()
    for box, cid, score in zip(boxes, classes, scores):
        cid = int(cid)
        x1, y1, x2, y2 = map(float, box)
        color = COLORS[cid]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, size // 5))
        label = f'{NAMES[cid]} {score:.0%}'
        x, y = max(0, x1), max(0, y1 - size - 8)
        rect = draw.textbbox((x, y), label, font=font)
        draw.rectangle((rect[0]-2, rect[1]-2, rect[2]+3, rect[3]+3), fill=color)
        draw.text((x, y), label, font=font, fill='white')
    return result


metrics = ROOT / "metrics.json"

if not metrics.is_file() or not (ROOT / "training_config.json").is_file():
    st.error("Missing metrics.json or training_config.json")
    st.stop()

try:
    ckpt = download_checkpoint()
except Exception as exc:
    st.error(f"Could not download model checkpoint: {exc}")
    st.stop()

if not 0.01 <= threshold <= 1 or cap < 1:
        raise ValueError('Invalid threshold or detection cap.')
except (KeyError, ValueError, TypeError) as exc:
    st.error(f'Could not read the evaluation settings: {exc}')
    st.stop()

with st.expander('Saved model evaluation · RF-DETR-M'):
    rows = []
    for split in ['valid', 'test']:
        values = config[split]
        rows.append({'Split': split, 'Precision': f"{values['precision']:.2%}",
                     'Recall': f"{values['recall']:.2%}", 'F1': f"{values['f1']:.2%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f'Fixed confidence: {threshold:.2f} · Detection cap: {cap} · Evaluation IoU: {config["iou"]:.2f}. These are dataset evaluation scores, not scores for the uploaded image.')

st.subheader('Assess a waste load')
st.caption('1 · Upload an image     2 · Record arrival time     3 · Review the action')
now = datetime.now(TZ)
with st.form('load_assessment_form'):
    uploaded = st.file_uploader('Waste image', type=['jpg','jpeg','png'])
    c1, c2 = st.columns(2)
    arrival_date = c1.date_input('Load arrival date', value=now.date())
    arrival_time = c2.time_input('Load arrival time · Riyadh', value=now.time().replace(second=0, microsecond=0, tzinfo=None))
    submitted = st.form_submit_button('Analyze load →', type='primary', use_container_width=True)

if submitted:
    if uploaded is None:
        st.warning('Upload a waste image first.')
    else:
        try:
            im = ImageOps.exif_transpose(Image.open(uploaded)).convert('RGB')
            arrival = datetime.combine(arrival_date, arrival_time, tzinfo=TZ)
            with st.spinner('Detecting materials and assessing the load…'):
                model, lock = load_model(str(ckpt), ckpt.stat().st_mtime_ns)
                with lock, torch.inference_mode():
                    d = model.predict(im, threshold=0.01)
                order = np.argsort(-d.confidence)[:cap]
                order = order[d.confidence[order] >= threshold]
                boxes, classes, scores = d.xyxy[order], d.class_id[order], d.confidence[order]
                if any(int(c) not in range(6) for c in classes):
                    raise ValueError('Checkpoint class IDs do not match the six-class waste dataset.')
                result = draw_boxes(im, boxes, classes, scores)
            st.session_state['load_assessment_result'] = dict(original=im, result=result, classes=classes,
                                                  scores=scores, arrival=arrival)
        except Exception as exc:
            st.session_state.pop('load_assessment_result', None)
            st.error(f'Analysis could not finish: {exc}')

if 'load_assessment_result' in st.session_state:
    r = st.session_state['load_assessment_result']
    classes = r['classes']
    total = len(classes)
    organic = int((classes == 0).sum())
    share = organic / total if total else 0
    st.divider()
    st.subheader('Load assessment')
    a, b, c = st.columns(3)
    a.metric('Detected items', total)
    b.metric('Organic items', organic)
    c.metric('Organic share', f'{share:.1%}' if total else '—')
    st.caption('Organic share is based on detected item counts, not weight or image area.')
    arrival = r['arrival']
    if not total:
        decision = 'MANUAL INSPECTION'
        deadline_text = '—'
        action = 'No items detected above the fixed confidence threshold. Inspect the load manually.'
        st.warning(action)
    elif share >= 0.30:
        decision = 'ORGANIC DIVERSION'
        deadline = arrival + timedelta(hours=24)
        deadline_text = deadline.strftime('%d %b %Y, %H:%M') + ' · Riyadh'
        action = 'Separate organic waste now and send it to composting or biogas within 24 hours of arrival.'
        st.error(f'🔴 {decision} — {action}')
        remaining = deadline - datetime.now(TZ)
        if remaining.total_seconds() < 0:
            st.warning('The 24-hour dispatch deadline has passed.')
        else:
            st.caption(f'Time remaining: {remaining.total_seconds()/3600:.1f} hours')
    else:
        decision = 'SCHEDULED RECYCLING'
        deadline_text = 'Await facility dispatch schedule'
        counts = np.bincount(classes.astype(int), minlength=6)
        dominant = NAMES[1 + int(np.argmax(counts[1:]))]
        action = f'Hold the load for scheduled recycling. Most detected non-organic items: {dominant}.'
        st.success(f'🟢 {decision} — {action}')
    st.dataframe(pd.DataFrame([{
        'Arrival': arrival.strftime('%d %b %Y, %H:%M') + ' · Riyadh',
        'Decision': decision, 'Dispatch deadline': deadline_text,
    }]), hide_index=True, use_container_width=True)
    left, right = st.columns(2)
    left.image(r['original'], caption='Original image', use_container_width=True)
    right.image(r['result'], caption='Detected materials', use_container_width=True)
    with st.expander('Material breakdown'):
        st.dataframe(pd.DataFrame({
            'Material': NAMES,
            'Detected items': [int((classes == cid).sum()) for cid in range(6)],
        }), hide_index=True, use_container_width=True)
    output = BytesIO()
    r['result'].save(output, format='PNG')
    st.download_button('Download result image', output.getvalue(), 'waste_assessment.png', 'image/png')
st.caption('Project decision rule: organic share ≥ 30% triggers diversion within 24 hours. Recycling dates require the facility schedule.')
