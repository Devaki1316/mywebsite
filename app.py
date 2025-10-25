import os, json, uuid, datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ✅ Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # this will automatically load your MAIL_, TWILIO_ values

# ----- CONFIG -----
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Mail config (reads from .env)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

db = SQLAlchemy(app)
mail = Mail(app)

# Twilio setup (reads from .env)
TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_AUTH = os.getenv('TWILIO_AUTH')
TWILIO_FROM = os.getenv('TWILIO_FROM')
USE_TWILIO = all([TWILIO_SID, TWILIO_AUTH, TWILIO_FROM])

if USE_TWILIO:
    from twilio.rest import Client
    twilio_client = Client(TWILIO_SID, TWILIO_AUTH)

# ----- LOGIN MANAGER -----
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
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    type = db.Column(db.String(10))  # lost or found
    name = db.Column(db.String(120))
    description = db.Column(db.Text)
    location = db.Column(db.String(120))
    contact = db.Column(db.String(120))
    image_filename = db.Column(db.String(200))
    features = db.Column(db.Text)
    date = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def feature_array(self):
        return np.array(json.loads(self.features), dtype=float)

# ✅ Notification model
class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    lost_item_id = db.Column(db.Integer, db.ForeignKey('item.id'))
    found_item_id = db.Column(db.Integer, db.ForeignKey('item.id'))
    message = db.Column(db.Text)
    sent_time = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    status = db.Column(db.String(20), default='sent')

with app.app_context():
    db.create_all()

# ----- MODEL LOAD -----
print("Loading MobileNetV2...")
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
print("Model loaded successfully.")

# ----- HELPER FUNCTIONS -----
def extract_features(img_path):
    img = kimage.load_img(img_path, target_size=(224,224))
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feats = model.predict(x)
    return feats.flatten().tolist()

def save_image(file_storage):
    ext = os.path.splitext(file_storage.filename)[1]
    unique_name = f"{uuid.uuid4()}{ext}"
    path = os.path.join(UPLOAD_FOLDER, unique_name)
    file_storage.save(path)
    return unique_name, path

def compute_similarity(a, b):
    try:
        return float(cosine_similarity([a], [b])[0][0])
    except Exception:
        return 0.0

# ✅ Combined notification sender (email + SMS)
def notify_user(lost_item, found_item, similarity):
    """Send email and/or SMS notification about a match."""
    user = User.query.get(lost_item.user_id)
    if not user:
        return

    message_text = (
        f"Hello {user.username},\n\n"
        f"We found a possible match for your lost item:\n"
        f"- Lost Item: {lost_item.name}\n"
        f"- Found Item: {found_item.name}\n"
        f"- Location: {found_item.location}\n"
        f"- Finder Contact: {found_item.contact}\n"
        f"- Similarity Score: {round(similarity, 3)}\n\n"
        f"Please get in touch to verify!"
    )

    # Save notification
    notif = Notification(receiver_id=user.id, lost_item_id=lost_item.id,
                         found_item_id=found_item.id, message=message_text)
    db.session.add(notif)
    db.session.commit()

    # Send Email
    if mail and user.email and app.config.get('MAIL_USERNAME'):
        try:
            msg = Message(
                subject="Match Found for Your Lost Item",
                sender=app.config['MAIL_USERNAME'],
                recipients=[user.email],
                body=message_text
            )
            mail.send(msg)
            print(f"✅ Email sent to {user.email}")
        except Exception as e:
            print("❌ Email send failed:", e)

    # Send SMS
    if USE_TWILIO and user.phone:
        try:
            twilio_client.messages.create(
                to=user.phone,
                from_=TWILIO_FROM,
                body=f"Match found for your lost item '{lost_item.name}'. "
                     f"Contact: {found_item.contact}, Location: {found_item.location}"
            )
            print(f"✅ SMS sent to {user.phone}")
        except Exception as e:
            print("❌ SMS send failed:", e)

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
        user = User(username=username, email=email, password=password, phone=phone)
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
        user = User.query.filter_by(email=email, password=password).first()
        if user:
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
    notifications = Notification.query.filter_by(receiver_id=current_user.id).order_by(Notification.sent_time.desc()).all()
    return render_template('dashboard.html', lost=lost, found=found, notifications=notifications)

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
        found_item = Item(user_id=current_user.id, type='found',
                          name=request.form.get('name',''),
                          description=request.form.get('description',''),
                          location=request.form.get('location',''),
                          contact=request.form.get('contact',''),
                          image_filename=filename,
                          features=json.dumps(feats))
        db.session.add(found_item)
        db.session.commit()

        matches = []
        lost_items = Item.query.filter_by(type='lost').all()
        for li in lost_items:
            sim = compute_similarity(np.array(feats), li.feature_array())
            if sim >= 0.75:
                matches.append({'item': li, 'similarity': round(sim,3)})
                notify_user(li, found_item, sim)

        return render_template('results.html', matches=matches, query_image=filename)
    return render_template('found.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/reset')
def reset():
    for f in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        except:
            pass
    db.drop_all()
    db.create_all()
    return "Reset complete"

if __name__ == '__main__':
    app.run(debug=True)