"""
╔══════════════════════════════════════════════════════════════╗
║  MODULE 3 — Bi-LSTM  +  NLP Pipeline  +  Live Frontend      ║
║  Input  : cnn_features.npz  (from Module 2)                  ║
║  Output : best_lstm.keras  +  served HTML app                ║
╚══════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════
#  PART A — LSTM TRAINING
# ═══════════════════════════════════════════════════════════════

# ─── CELL 1 : Imports ────────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling1D, LayerNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json, re, os

print("TF:", tf.__version__)
tf.keras.backend.clear_session()

# ─── CELL 2 : Load CNN embeddings ────────────────────────────────────────────
"""
CNN output shape : (N_clips, 256)
For the LSTM we need sequences.  Since each WLASL clip is already a
temporal sequence that got condensed by GlobalAvgPool, we reshape:
  single embedding (256,)  →  fake sequence of length 1

A better approach (used here if cnn_features has the time axis preserved):
  Use the Conv1D output BEFORE GlobalAvgPool  →  (seq_reduced, 256)
  This gives the LSTM real temporal information.

Both variants are handled below.
"""
data    = np.load("/content/wlasl/cnn_features.npz")
E_train = data["E_train"]   # (N, 256) or (N, T, 256)
E_val   = data["E_val"]
E_test  = data["E_test"]
y_train = data["y_train"].astype("int32")
y_val   = data["y_val"].astype("int32")
y_test  = data["y_test"].astype("int32")

with open("/content/wlasl/class_map.json") as f:
    cmap      = json.load(f)
    IDX2CLASS = {int(k): v for k, v in cmap["idx2class"].items()}

NUM_CLASSES = len(IDX2CLASS)
print(f"Embedding shape : {E_train.shape}")
print(f"Classes         : {NUM_CLASSES}")

# If embeddings are 2-D (N, dim) add a time axis so LSTM gets (N, 1, dim)
if E_train.ndim == 2:
    E_train = E_train[:, np.newaxis, :]   # (N, 1, 256)
    E_val   = E_val[:,   np.newaxis, :]
    E_test  = E_test[:,  np.newaxis, :]

SEQ_LEN  = E_train.shape[1]
EMB_DIM  = E_train.shape[2]
print(f"LSTM input shape: ({SEQ_LEN}, {EMB_DIM})")

# ─── CELL 3 : Build Bi-LSTM ───────────────────────────────────────────────────
def build_lstm(seq_len, emb_dim, num_classes):
    inp = Input(shape=(seq_len, emb_dim), name="embedding_input")

    x = Bidirectional(LSTM(256, return_sequences=True))(inp)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    x   = Dense(256, activation="relu")(x)
    x   = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax", name="output")(x)

    return Model(inputs=inp, outputs=out, name="ASL_BiLSTM")


lstm_model = build_lstm(SEQ_LEN, EMB_DIM, NUM_CLASSES)
lstm_model.compile(
    optimizer = tf.keras.optimizers.Adam(1e-3),
    loss      = "sparse_categorical_crossentropy",
    metrics   = ["accuracy"]
)
lstm_model.summary()

# ─── CELL 4 : Train LSTM ─────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
    ModelCheckpoint("/content/wlasl/best_lstm.keras",
                    save_best_only=True, monitor="val_accuracy"),
    ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, verbose=1)
]

history = lstm_model.fit(
    E_train, y_train,
    validation_data = (E_val, y_val),
    epochs          = 60,
    batch_size      = 32,
    callbacks       = callbacks,
    verbose         = 1
)

loss, acc = lstm_model.evaluate(E_test, y_test, verbose=0)
print(f"\n🎯 LSTM Test Accuracy : {acc * 100:.2f}%")
lstm_model.save("/content/wlasl/sign_language_lstm_final.keras")

# ═══════════════════════════════════════════════════════════════
#  PART B — NLP PIPELINE
# ═══════════════════════════════════════════════════════════════

# ─── CELL 5 : NLP helpers ────────────────────────────────────────────────────
def remove_duplicates(words):
    """AAABBB → AB"""
    if not words:
        return []
    out = [words[0]]
    for w in words[1:]:
        if w != out[-1]:
            out.append(w)
    return out


# Common ASL-to-English expansions
EXPANSIONS = {
    "YOU"    : "you",   "I"   : "I",    "HELP": "help",
    "WANT"   : "want",  "NEED": "need", "LIKE": "like",
    "GOOD"   : "good",  "BAD" : "bad",  "GO"  : "go",
    "COME"   : "come",  "STOP": "stop", "YES" : "yes",
    "NO"     : "no",    "WHAT": "what", "WHERE": "where",
    "WHEN"   : "when",  "HOW" : "how",  "WHO" : "who",
    "PLEASE" : "please","THANK": "thank","YOU": "you",
    "NAME"   : "name",  "MY"  : "my",   "YOUR": "your",
    "HELLO"  : "hello", "BYE" : "bye",  "SORRY": "sorry",
    "MORE"   : "more",  "AGAIN": "again","WHERE":"where",
    "FOOD"   : "food",  "WATER":"water","EAT" : "eat",
    "DRINK"  : "drink", "WORK": "work", "SCHOOL":"school",
    "HOME"   : "home",  "LOVE": "love", "FRIEND":"friend",
    "FAMILY" : "family","TIME":"time",  "DAY" : "day",
    "NIGHT"  : "night", "MORNING":"morning",
}


def signs_to_sentence(sign_list):
    """
    sign_list : list of predicted gloss strings  e.g. ['HELLO', 'YOU', 'NAME']
    Returns   : readable English sentence
    """
    # 1. de-duplicate consecutive repeats
    signs = remove_duplicates(sign_list)

    # 2. expand each gloss
    words = [EXPANSIONS.get(s.upper(), s.lower()) for s in signs]

    # 3. join
    sentence = " ".join(words).strip()

    # 4. capitalise first letter + ensure punctuation
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
        if sentence[-1] not in ".!?":
            sentence += "."

    return sentence


def nlp_pipeline(pred_indices):
    """End-to-end: index list → corrected English sentence"""
    glosses  = [IDX2CLASS[i] for i in pred_indices]
    sentence = signs_to_sentence(glosses)
    print(f"  Glosses   : {glosses}")
    print(f"  Sentence  : {sentence}")
    return sentence


# Quick smoke-test
print("\n--- NLP smoke test ---")
nlp_pipeline([0, 1, 2])

# ═══════════════════════════════════════════════════════════════
#  PART C — LIVE FRONTEND (served via Flask + localtunnel)
# ═══════════════════════════════════════════════════════════════

# ─── CELL 6 : Install tunnel ─────────────────────────────────────────────────
!pip install flask -q
!npm install -g localtunnel -q

# ─── CELL 7 : Write the HTML app ─────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SignSpeak — Live ASL</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
<style>
  :root{
    --bg:#0c0c10;--surface:#13131a;--surface2:#1c1c26;
    --accent:#39ff85;--accent2:#9d4edd;--accent3:#ff6b35;
    --text:#e8e8f0;--muted:#55556a;--border:#2a2a38;
    --font-head:"Syne",sans-serif;--font-mono:"DM Mono",monospace;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%;background:var(--bg);color:var(--text);
    font-family:var(--font-head);overflow-x:hidden}

  /* ── Grain overlay ── */
  body::before{content:"";position:fixed;inset:0;
    background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
    pointer-events:none;z-index:9999;opacity:.4}

  /* ── Layout ── */
  .shell{max-width:1200px;margin:0 auto;padding:24px 20px;
    display:grid;grid-template-rows:auto 1fr auto;min-height:100vh;gap:24px}

  /* ── Header ── */
  header{display:flex;align-items:center;justify-content:space-between;
    border-bottom:1px solid var(--border);padding-bottom:16px}
  .logo{font-size:1.6rem;font-weight:800;letter-spacing:-.02em}
  .logo span{color:var(--accent)}
  .badge{font-size:.75rem;font-family:var(--font-mono);padding:4px 10px;
    border:1px solid var(--accent);color:var(--accent);border-radius:4px;
    animation:pulse 2s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

  /* ── Main grid ── */
  .main{display:grid;grid-template-columns:1fr 320px;gap:20px;align-items:start}
  @media(max-width:820px){.main{grid-template-columns:1fr}}

  /* ── Camera card ── */
  .cam-card{position:relative;border-radius:18px;overflow:hidden;
    border:1.5px solid var(--border);background:var(--surface)}
  #output_canvas{width:100%;display:block;border-radius:18px}
  .cam-overlay{position:absolute;bottom:0;left:0;right:0;
    padding:12px 16px;background:linear-gradient(transparent,rgba(12,12,16,.9));
    font-size:.8rem;font-family:var(--font-mono);color:var(--muted)}
  .cam-overlay .status-dot{display:inline-block;width:7px;height:7px;
    border-radius:50%;background:var(--accent);margin-right:6px;
    animation:pulse 1.5s infinite}

  /* ── Right panel ── */
  .panel{display:flex;flex-direction:column;gap:16px}

  .card{background:var(--surface);border:1.5px solid var(--border);
    border-radius:16px;padding:20px}
  .card-label{font-size:.7rem;font-family:var(--font-mono);
    color:var(--muted);letter-spacing:.12em;text-transform:uppercase;
    margin-bottom:12px}

  /* ── Sign display ── */
  .sign-big{font-size:5rem;font-weight:800;line-height:1;
    color:var(--accent);text-shadow:0 0 40px rgba(57,255,133,.35);
    transition:all .15s ease;text-align:center;min-height:6rem;
    display:flex;align-items:center;justify-content:center}

  /* ── Bars ── */
  .bar-wrap{background:var(--bg);border-radius:8px;height:10px;
    overflow:hidden;margin-top:10px}
  .bar-fill{height:100%;border-radius:8px;transition:width .2s ease;width:0%}
  .bar-conf{background:linear-gradient(90deg,var(--accent),#00d4ff)}
  .bar-hold{background:var(--accent3)}
  .bar-label{font-size:.7rem;font-family:var(--font-mono);
    color:var(--muted);margin-top:4px}

  /* ── History chips ── */
  .chip-row{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
  .chip{font-family:var(--font-mono);font-size:.85rem;
    background:var(--surface2);border:1px solid var(--border);
    padding:3px 10px;border-radius:6px;color:var(--text)}

  /* ── Text output ── */
  .text-section{grid-column:1/-1}
  .text-box{background:var(--surface);border:1.5px solid var(--accent);
    border-radius:14px;padding:20px 24px;
    font-family:var(--font-mono);font-size:1.4rem;letter-spacing:.06em;
    min-height:70px;word-break:break-word;line-height:1.5;
    position:relative;color:var(--text)}
  .cursor{display:inline-block;width:2px;height:1.4em;
    background:var(--accent);margin-left:3px;
    vertical-align:middle;animation:blink 1s infinite}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

  /* ── Buttons ── */
  .btn-row{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px}
  .btn{font-family:var(--font-head);font-weight:700;font-size:.85rem;
    padding:10px 20px;border:none;border-radius:10px;cursor:pointer;
    transition:transform .1s,filter .15s;letter-spacing:.02em}
  .btn:hover{filter:brightness(1.15);transform:translateY(-1px)}
  .btn:active{transform:scale(.97)}
  .btn-red  {background:#c0392b;color:#fff}
  .btn-blue {background:#2980b9;color:#fff}
  .btn-amber{background:#d68910;color:#fff}
  .btn-purple{background:var(--accent2);color:#fff}
  .btn-green{background:#1e8449;color:#fff}

  /* ── NLP output ── */
  .nlp-box{background:var(--surface2);border-left:3px solid var(--accent2);
    border-radius:0 12px 12px 0;padding:14px 18px;
    font-size:1rem;color:var(--text);min-height:46px;
    font-family:var(--font-mono);word-break:break-word}
  .nlp-label{font-size:.7rem;font-family:var(--font-mono);
    color:var(--accent2);letter-spacing:.1em;text-transform:uppercase;
    margin-bottom:6px}
</style>
</head>
<body>
<div class="shell">

  <!-- ── Header ── -->
  <header>
    <div class="logo">Sign<span>Speak</span></div>
    <div class="badge">● LIVE ASL · WLASL-100</div>
  </header>

  <!-- ── Main ── -->
  <div class="main">

    <!-- Camera -->
    <div>
      <div class="cam-card">
        <video id="input_video" style="display:none"></video>
        <canvas id="output_canvas" width="640" height="480"></canvas>
        <div class="cam-overlay">
          <span class="status-dot"></span>
          <span id="status_text">Initialising camera…</span>
        </div>
      </div>
    </div>

    <!-- Right panel -->
    <div class="panel">

      <!-- Current sign -->
      <div class="card">
        <div class="card-label">Detected Sign</div>
        <div class="sign-big" id="sign_display">–</div>
        <div class="bar-wrap"><div class="bar-fill bar-conf" id="conf_fill"></div></div>
        <div class="bar-label" id="conf_label">Confidence —</div>
        <div class="bar-wrap" style="margin-top:8px"><div class="bar-fill bar-hold" id="hold_fill"></div></div>
        <div class="bar-label" id="hold_label">Hold to commit…</div>
      </div>

      <!-- History -->
      <div class="card">
        <div class="card-label">Last 12 Signs</div>
        <div class="chip-row" id="history_list"></div>
      </div>

      <!-- NLP sentence -->
      <div class="card">
        <div class="nlp-label">NLP → English</div>
        <div class="nlp-box" id="nlp_out">Predicted sentence appears here…</div>
      </div>

    </div><!-- /panel -->

    <!-- Full-width text output -->
    <div class="text-section">
      <div class="card-label" style="margin-bottom:8px">Raw Signs → Text</div>
      <div class="text-box">
        <span id="text_content"></span><span class="cursor"></span>
      </div>
      <div class="btn-row">
        <button class="btn btn-red"    onclick="clearText()">🗑 Clear</button>
        <button class="btn btn-blue"   onclick="addSpace()">␣ Space</button>
        <button class="btn btn-amber"  onclick="deleteLast()">⌫ Delete</button>
        <button class="btn btn-purple" onclick="copyText()">📋 Copy</button>
        <button class="btn btn-green"  onclick="runNLP()">✦ NLP Sentence</button>
      </div>
    </div>

  </div><!-- /main -->
</div><!-- /shell -->

<script>
/* ═══════════════════════════════════════════════════════════
   Client-side classifier
   Uses MediaPipe Holistic (pose + both hands) to extract
   landmarks and run a rule-based ASL classifier.
   In production this would call a backend inference endpoint.
═══════════════════════════════════════════════════════════ */

const video     = document.getElementById("input_video");
const canvas    = document.getElementById("output_canvas");
const ctx       = canvas.getContext("2d");
const signDisp  = document.getElementById("sign_display");
const textEl    = document.getElementById("text_content");
const confFill  = document.getElementById("conf_fill");
const holdFill  = document.getElementById("hold_fill");
const confLabel = document.getElementById("conf_label");
const holdLabel = document.getElementById("hold_label");
const statusTxt = document.getElementById("status_text");
const histList  = document.getElementById("history_list");
const nlpOut    = document.getElementById("nlp_out");

const off    = document.createElement("canvas");
off.width=640; off.height=480;
const offCtx = off.getContext("2d");

let rawText="", lastSign="", holdFrames=0, lastAdded="", history=[];
const HOLD_THRESHOLD = 18;

// ── Text controls ────────────────────────────────────────────
function updateText(){ textEl.innerText = rawText; }
function clearText() { rawText=""; updateText(); history=[]; renderHistory(); nlpOut.innerText=""; }
function addSpace()  { rawText+=" "; updateText(); }
function deleteLast(){ rawText=rawText.slice(0,-1); updateText(); }
function copyText()  {
  navigator.clipboard.writeText(rawText);
  statusTxt.innerText="✅ Copied to clipboard";
  setTimeout(()=>statusTxt.innerText="",2000);
}

// ── NLP (client-side rule-based mirror of Python pipeline) ──
const EXPAND = {
  YOU:"you",I:"I",HELP:"help",WANT:"want",NEED:"need",LIKE:"like",
  GOOD:"good",BAD:"bad",GO:"go",COME:"come",STOP:"stop",YES:"yes",
  NO:"no",WHAT:"what",WHERE:"where",WHEN:"when",HOW:"how",WHO:"who",
  PLEASE:"please",THANK:"thank",NAME:"name",MY:"my",YOUR:"your",
  HELLO:"hello",BYE:"bye",SORRY:"sorry",MORE:"more",AGAIN:"again",
  FOOD:"food",WATER:"water",EAT:"eat",DRINK:"drink",WORK:"work",
  SCHOOL:"school",HOME:"home",LOVE:"love",FRIEND:"friend",
  FAMILY:"family",TIME:"time",DAY:"day",NIGHT:"night",MORNING:"morning",
};

function nlpSentence(signs){
  const deduped = signs.filter((s,i)=> i===0 || s!==signs[i-1]);
  const words   = deduped.map(s=> EXPAND[s.toUpperCase()] || s.toLowerCase());
  let sent = words.join(" ").trim();
  if(!sent) return "";
  sent = sent[0].toUpperCase() + sent.slice(1);
  if(!".!?".includes(sent.slice(-1))) sent += ".";
  return sent;
}

function runNLP(){
  const signs = rawText.trim().split(" ").filter(Boolean);
  if(!signs.length){ nlpOut.innerText="Nothing to translate yet."; return; }
  nlpOut.innerText = nlpSentence(signs) || rawText;
}

// ── History chips ────────────────────────────────────────────
function addToHistory(s){
  history.push(s);
  if(history.length>12) history.shift();
  renderHistory();
}
function renderHistory(){
  histList.innerHTML = history.map(s=>`<span class="chip">${s}</span>`).join("");
}

// ── Geometry helpers ──────────────────────────────────────────
function dist3(a,b){ return Math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2); }
function up(tip,pip){ return tip.y<pip.y; }

// ── ASL hand classifier (right-hand landmarks) ───────────────
function classifyHand(L){
  if(!L||L.length<21) return{sign:"?",conf:0};
  try{
    const tTip=L[4],tIP=L[3],iTip=L[8],iPIP=L[6],iMCP=L[5],
          mTip=L[12],mPIP=L[10],rTip=L[16],rPIP=L[14],
          pTip=L[20],pPIP=L[18],wrist=L[0];

    const iUp=up(iTip,iPIP), mUp=up(mTip,mPIP),
          rUp=up(rTip,rPIP), pUp=up(pTip,pPIP);

    const thumbOut = dist3(tTip,iTip)>0.15;
    const allCurl  = !iUp&&!mUp&&!rUp&&!pUp;
    const allUp    =  iUp&& mUp&& rUp&& pUp;
    const pinch    = dist3(tTip,iTip);

    if(allCurl&&thumbOut&&tTip.y>tIP.y)                           return{sign:"A",conf:90};
    if(allUp&&!thumbOut)                                           return{sign:"B",conf:87};
    if(!allCurl&&!allUp&&!iUp&&!pUp&&pinch<0.22&&dist3(tTip,pTip)<0.30) return{sign:"C",conf:83};
    if(iUp&&!mUp&&!rUp&&!pUp&&dist3(tTip,mTip)<0.07)             return{sign:"D",conf:82};
    if(allCurl&&!thumbOut&&pinch<0.08)                            return{sign:"E",conf:80};
    if(!iUp&&mUp&&rUp&&pUp&&pinch<0.06)                          return{sign:"F",conf:82};
    if(!iUp&&!mUp&&!rUp&&pUp&&!thumbOut)                         return{sign:"I",conf:84};
    if(iUp&&!mUp&&!rUp&&!pUp&&thumbOut)                          return{sign:"L",conf:90};
    if(!allUp&&!allCurl&&pinch<0.07&&dist3(mTip,tTip)<0.09)      return{sign:"O",conf:80};
    if(iUp&&mUp&&!rUp&&!pUp&&!thumbOut&&Math.abs(iTip.x-mTip.x)<0.04) return{sign:"R",conf:76};
    if(allCurl&&!thumbOut&&tTip.y<iPIP.y)                        return{sign:"S",conf:80};
    if(iUp&&mUp&&!rUp&&!pUp&&!thumbOut&&dist3(iTip,mTip)<0.06)  return{sign:"U",conf:80};
    if(iUp&&mUp&&!rUp&&!pUp&&!thumbOut&&dist3(iTip,mTip)>0.07)  return{sign:"V",conf:86};
    if(iUp&&mUp&&rUp&&!pUp)                                      return{sign:"W",conf:84};
    if(!iUp&&!mUp&&!rUp&&pUp&&thumbOut)                          return{sign:"Y",conf:86};
    if(allUp&&thumbOut)                                           return{sign:"5",conf:82};
  }catch(e){}
  return{sign:"?",conf:0};
}

// ── MediaPipe Holistic results handler ───────────────────────
function onResults(results){
  try{
    offCtx.save();
    offCtx.clearRect(0,0,640,480);
    offCtx.translate(640,0); offCtx.scale(-1,1);
    offCtx.drawImage(results.image,0,0,640,480);
    offCtx.restore();

    const rh = results.rightHandLandmarks;
    const lh = results.leftHandLandmarks;
    const pose = results.poseLandmarks;

    if(rh){
      offCtx.save(); offCtx.translate(640,0); offCtx.scale(-1,1);
      drawConnectors(offCtx,rh,HAND_CONNECTIONS,{color:"#39ff85",lineWidth:2});
      drawLandmarks(offCtx,rh,{color:"#ff4444",radius:4});
      offCtx.restore();
    }
    if(lh){
      offCtx.save(); offCtx.translate(640,0); offCtx.scale(-1,1);
      drawConnectors(offCtx,lh,HAND_CONNECTIONS,{color:"#9d4edd",lineWidth:2});
      drawLandmarks(offCtx,lh,{color:"#ff9944",radius:4});
      offCtx.restore();
    }
    if(pose){
      offCtx.save(); offCtx.translate(640,0); offCtx.scale(-1,1);
      drawConnectors(offCtx,pose,POSE_CONNECTIONS,{color:"rgba(255,255,255,.15)",lineWidth:1});
      offCtx.restore();
    }

    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(off,0,0);

    // Classify (prefer right hand, fall back to left)
    const {sign,conf} = classifyHand(rh || lh);

    signDisp.innerText   = sign;
    confFill.style.width = conf+"%";
    confLabel.innerText  = `Confidence — ${conf}%`;

    if(sign===lastSign && sign!=="?"){
      holdFrames++;
      const pct = Math.min(100,(holdFrames/HOLD_THRESHOLD)*100);
      holdFill.style.width = pct+"%";
      holdLabel.innerText  = holdFrames>=HOLD_THRESHOLD ? `✅ "${sign}" committed!` : `Hold… ${Math.round(pct)}%`;
      if(holdFrames===HOLD_THRESHOLD && sign!==lastAdded){
        rawText  += sign;
        updateText();
        addToHistory(sign);
        lastAdded = sign;
      }
    } else {
      lastSign   = sign;
      holdFrames = 0;
      holdFill.style.width="0%";
      holdLabel.innerText="Hold to commit…";
      if(sign!==lastAdded) lastAdded="";
    }

    statusTxt.innerText = (rh||lh)
      ? (sign==="?" ? "🤔 Sign not recognised" : `Detecting: ${sign}`)
      : "👋 Show a hand sign";

  }catch(err){ console.warn("Frame error:",err); }
}

// ── Start holistic ───────────────────────────────────────────
const holistic = new Holistic({
  locateFile: f=>`https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`
});
holistic.setOptions({
  modelComplexity:1,
  smoothLandmarks:true,
  enableSegmentation:false,
  minDetectionConfidence:0.6,
  minTrackingConfidence:0.5
});
holistic.onResults(onResults);

const camera = new Camera(video,{
  onFrame: async()=>{ try{ await holistic.send({image:video}); }catch(e){} },
  width:640, height:480
});
camera.start()
  .then(()=> statusTxt.innerText="✅ Camera live — show a hand sign!")
  .catch(e => statusTxt.innerText="❌ "+e.message);
</script>
</body>
</html>
"""

# ─── CELL 8 : Save HTML & serve ──────────────────────────────────────────────
import threading, subprocess, time
from flask import Flask

with open("/content/asl_live.html", "w") as f:
    f.write(HTML)

app = Flask(__name__)

@app.route("/")
def index():
    with open("/content/asl_live.html") as f:
        return f.read()

threading.Thread(
    target=lambda: app.run(port=5000, use_reloader=False),
    daemon=True
).start()

time.sleep(2)

proc = subprocess.Popen(["lt", "--port", "5000"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(3)
line = proc.stdout.readline().decode().strip()
print(f"\n✅ Open this link in your browser:\n\n   👉  {line}  👈\n")
print("Grant camera access → show a hand sign → hold steady to commit!")
print("\n💡 Your public IP (if localtunnel asks):")
os.system("curl -s ifconfig.me")
