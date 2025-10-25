import os, json, uuid, datetime
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import generate_password_hash, check_password_hash

# Suppress TF GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----- CONFIG -----
app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Ensure uploads folder exists
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Mail config
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')

db = SQLAlchemy(app)
mail = Mail(app)

# Twilio optional
TWILIO_SID = os.environ.get('TWILIO_SID')
TWILIO_AUTH = os.environ.get('TWILIO_AUTH')
TWILIO_FROM = os.environ.get('TWILIO_FROM')
USE_TWILIO = all([TWILIO_SID, TWILIO_AUTH, TWILIO_FROM])
if USE_TWILIO:
    from twilio.rest import Client
    twilio_client = Client(TWILIO_SID, TWILIO_AUTH)

# Login manager
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ----- MODELS -----
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    phone = db.Column(db.String(30), nullable=True)

class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    type = db.Column(db.String(10))  # 'lost' or 'found'
    name = db.Column(db.String(120))
    description = db.Column(db.Text)
    location = db.Column(db.String(120))
    contact = db.Column(db.String(120))
    image_filename = db.Column(db.String(200))
    features = db.Column(db.Text)
    date = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def feature_array(self):
        return np.array(json.loads(self.features), dtype=float)

with app.app_context():
    db.create_all()

# ----- MODEL LOAD -----
print("Loading MobileNetV2 (this may take a few seconds)...")
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
print("Model loaded.")

def extract_features(img_path):
    img = kimage.load_img(img_path, target_size=(224,224))
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feats = model.predict(x)
    return feats.flatten().tolist()

def save_image(file_storage):
    ext = os.path.splitext(file_storage.filename)[1]
    unique_name = str(uuid.uuid4()) + ext
    path = os.path.join(UPLOAD_FOLDER, unique_name)
    file_storage.save(path)
    return unique_name, path

def compute_similarity(a, b):
    try:
        return float(cosine_similarity([a], [b])[0][0])
    except:
        return 0.0

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ----- ROUTES -----
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip().lower()
        password = request.form['password']
        phone = request.form.get('phone','').strip()
        if User.query.filter((User.username==username)|(User.email==email)).first():
            flash("Username or email already exists.", "error")
            return redirect(url_for('register'))
        password_hash = generate_password_hash(password)
        user = User(username=username, email=email, password=password_hash, phone=phone)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash("Logged in successfully.", "success")
            return redirect(url_for('dashboard'))
        flash("Invalid credentials.", "error")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    lost = Item.query.filter_by(user_id=current_user.id, type='lost').all()
    found = Item.query.filter_by(user_id=current_user.id, type='found').all()
    return render_template('dashboard.html', lost=lost, found=found)

@app.route('/lost', methods=['GET','POST'])
@login_required
def lost():
    if request.method == 'POST':
        f = request.files.get('image')
        if not f:
            flash("Please upload an image.", "error")
            return redirect(url_for('lost'))
        filename, path = save_image(f)
        feats = extract_features(path)
        item = Item(user_id=current_user.id, type='lost', name=request.form.get('name',''),
                    description=request.form.get('description',''),
                    location=request.form.get('location',''),
                    contact=request.form.get('contact',''),
                    image_filename=filename, features=json.dumps(feats))
        db.session.add(item)
        db.session.commit()
        flash("Lost item reported.", "success")
        return redirect(url_for('dashboard'))
    return render_template('lost.html')

@app.route('/found', methods=['GET','POST'])
@login_required
def found():
    if request.method == 'POST':
        f = request.files.get('image')
        if not f:
            flash("Please upload an image.", "error")
            return redirect(url_for('found'))
        filename, path = save_image(f)
        feats = extract_features(path)
        found_item = Item(user_id=current_user.id, type='found', name=request.form.get('name',''),
                          description=request.form.get('description',''),
                          location=request.form.get('location',''),
                          contact=request.form.get('contact',''),
                          image_filename=filename, features=json.dumps(feats))
        db.session.add(found_item)
        db.session.commit()

        matches = []
        lost_items = Item.query.filter_by(type='lost').all()
        for li in lost_items:
            sim = compute_similarity(np.array(feats), li.feature_array())
            if sim >= 0.75:
                matches.append({'item': li, 'similarity': round(sim,3)})
        return render_template('results.html', matches=matches, query_image=filename)
    return render_template('found.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/reset')
def reset():
    for f in os.listdir(UPLOAD_FOLDER):
        try: os.remove(os.path.join(UPLOAD_FOLDER, f))
        except: pass
    db.drop_all()
    db.create_all()
    return "Reset complete"

# ----- RUN -----
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
