"""
MetaGuard - Network Intrusion Detection
"""
from flask import Flask, render_template, jsonify, request
import threading, time, os, joblib, random
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP, ICMP

app = Flask(__name__)
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
FEATURES = ['Init Fwd Win Byts', 'Fwd Seg Size Min', 'Protocol', 'Fwd Header Len', 'Fwd Pkt Len Max', 'ACK Flag Cnt']

sniffer_active = False
prediction_results = []
ensemble = scaler = None
# Deep learning NN integration (optional)
selected_model = 'ensemble'
nn_model_wrapper = None
nn_available = False

# Lightweight wrapper for NN model (if available)
class _NNModelWrapper:
    def __init__(self, path):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, num_classes)
                )
            def forward(self, x):
                return self.net(x)

        chk = torch.load(path, map_location='cpu')
        num_classes = len(chk['label_encoder'].classes_)
        model = Net(input_size=len(FEATURES), num_classes=num_classes)
        model.load_state_dict(chk['model_state_dict'])
        model.eval()
        self.model = model
        self.scaler = chk['scaler']
        self.label_encoder = chk['label_encoder']
        self.features = chk['features']
        self.device = torch.device('cpu')
        self.model.to(self.device)
    def predict(self, X):
        import torch
        Xs = self.scaler.transform(X)
        with torch.no_grad():
            out = self.model(torch.FloatTensor(Xs))
            idx = int(out.argmax(dim=1).item())
        return self.label_encoder.inverse_transform([idx])[0]

def _load_nn_model():
    global nn_model_wrapper, nn_available
    import os
    # Try common NN model path
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DeepLearning', 'results', 'nn_model.pth'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'DeepLearning', 'results', 'nn_model.pth')
    ]
    for p in candidates:
        if p and os.path.exists(p):
            try:
                nn_model_wrapper = _NNModelWrapper(p)
                nn_available = True
                return True
            except Exception as e:
                print(f"[NN] Failed to load NN model: {e}")
                return False
    return False

print("Loading model...")
ensemble = joblib.load(os.path.join(MODELS_DIR, 'realtime_ensemble.pkl'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'realtime_scaler.pkl'))
print("Loaded!")
_load_nn_model()
if nn_available:
    print("NN model loaded and ready for use.")
else:
    print("NN model not found; continuing with Ensemble only.")

def get_proto_name(proto_num):
    mapping = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}
    return mapping.get(proto_num, str(proto_num))

def get_features(pkt):
    if not pkt.haslayer(IP):
        return None
    try:
        f = {'Init Fwd Win Byts': 65535, 'Fwd Seg Size Min': 20, 'Protocol': 6,
             'Fwd Header Len': 20, 'Fwd Pkt Len Max': 64, 'ACK Flag Cnt': 0}
        if pkt.haslayer(TCP):
            f['Protocol'] = 6
            f['Init Fwd Win Byts'] = int(pkt[TCP].window) if pkt[TCP].window else 65535
            f['Fwd Header Len'] = int(pkt[TCP].dataofs * 4) if pkt[TCP].dataofs else 20
            f['ACK Flag Cnt'] = 1 if pkt[TCP].flags.A else 0
            f['Fwd Seg Size Min'] = int(pkt[TCP].dataofs * 4) if pkt[TCP].dataofs else 20
        elif pkt.haslayer(UDP):
            f['Protocol'] = 17
            f['Init Fwd Win Byts'] = 0
            f['Fwd Header Len'] = 8
        f['Fwd Pkt Len Max'] = int(len(pkt))
        return f
    except:
        return None

def do_predict(X):
    # X is expected as a 2D numpy array with shape (1, n_features)
    global nn_model_wrapper
    if nn_available and nn_model_wrapper is not None:
        try:
            return nn_model_wrapper.predict(X)
        except Exception as e:
            print(f"[NN] Prediction error: {e}")
    try:
        pred = ensemble.predict(X)[0]
        return pred
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown"

def process(pkt):
    global prediction_results, sniffer_active
    
    # Always process regardless of sniffer_active
    if not pkt.haslayer(IP):
        return
        
    try:
        # Get features
        f = {'Init Fwd Win Byts': 65535, 'Fwd Seg Size Min': 20, 'Protocol': 6,
             'Fwd Header Len': 20, 'Fwd Pkt Len Max': 64, 'ACK Flag Cnt': 0}
        if pkt.haslayer(TCP):
            f['Protocol'] = 6
            f['Init Fwd Win Byts'] = int(pkt[TCP].window) if pkt[TCP].window else 65535
            f['Fwd Header Len'] = int(pkt[TCP].dataofs * 4) if pkt[TCP].dataofs else 20
            f['ACK Flag Cnt'] = 1 if pkt[TCP].flags.A else 0
        elif pkt.haslayer(UDP):
            f['Protocol'] = 17
            f['Init Fwd Win Byts'] = 0
            f['Fwd Header Len'] = 8
        f['Fwd Pkt Len Max'] = int(len(pkt))
        
        # Create feature vector and predict (support NN or Ensemble)
        import numpy as np
        X_vec = np.array([[f[col] for col in FEATURES]], dtype=float)
        pred = do_predict(X_vec)
        
        # Get protocol name
        pn = int(pkt[IP].proto) if pkt.haslayer(IP) else 0
        proto = get_proto_name(pn)
        
        result = {
            'src': str(pkt[IP].src),
            'dst': str(pkt[IP].dst),
            'proto': proto,
            'len': len(pkt),
            'prediction': str(pred),
            'timestamp': time.strftime('%H:%M:%S')
        }
        
        prediction_results.append(result)
        print(f"[{result['timestamp']}] {result['proto']} {result['src']} -> {result['dst']} = {result['prediction']}")
        
        if len(prediction_results) > 100:
            prediction_results = prediction_results[-100:]
            
    except Exception as e:
        print(f"Error: {e}")

def sniffer_loop():
    global sniffer_active, prediction_results
    print("Sniffer thread started")
    while sniffer_active:
        try:
            sniff(prn=process, count=1, store=0, timeout=1)
        except Exception as e:
            print(f"Sniff error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    global sniffer_active, prediction_results
    global selected_model
    data = request.get_json(silent=True) or {}
    m = data.get('model', 'ensemble')
    if m in ('ensemble', 'nn'):
        selected_model = m
    else:
        selected_model = 'ensemble'
    if not sniffer_active:
        prediction_results = []
        sniffer_active = True
        t = threading.Thread(target=sniffer_loop, daemon=True)
        t.start()
        import time
        time.sleep(0.5)
        print("Sniffer thread started")
    return jsonify({'status': 'started'})

@app.route('/demo', methods=['POST'])
def demo():
    import random
    global prediction_results
    ips = ['192.168.1.10', '192.168.1.20', '10.0.0.5', '172.16.0.1']
    protos = ['TCP', 'UDP']
    labels = ['Benign', 'Benign', 'Benign', 'Benign', 'Bot', 'DDOS attack-HOIC', 'FTP-BruteForce']
    
    demo_results = []
    for i in range(20):
        demo_results.append({
            'src': random.choice(ips),
            'dst': random.choice(ips),
            'proto': random.choice(protos),
            'len': random.randint(60, 1500),
            'prediction': random.choice(labels),
            'timestamp': time.strftime('%H:%M:%S')
        })
    
    prediction_results = demo_results
    return jsonify({'status': 'demo', 'count': len(demo_results)})

@app.route('/debug')
def debug():
    return jsonify({'active': sniffer_active, 'count': len(prediction_results)})

@app.route('/stop', methods=['POST'])
def stop():
    global sniffer_active, prediction_results
    sniffer_active = False
    print("Sniffer stopped")
    return jsonify({'status': 'stopped'})

@app.route('/results')
def results():
    return jsonify({'predictions': prediction_results})

def cleanup():
    global sniffer_active
    sniffer_active = False
    print("\nShutting down...")

import signal
import sys

def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    try:
        app.run(debug=False, port=4000)
    except KeyboardInterrupt:
        cleanup()
    finally:
        cleanup()
