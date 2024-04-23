# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
import json
import os
from datetime import datetime

import PyPDF2
import docx
import pyttsx3
from flask import Flask, render_template, session, redirect, request, send_file, url_for
from flask_sqlalchemy import SQLAlchemy
from googletrans import Translator
from gtts import gTTS
from sqlalchemy import func
from werkzeug.security import generate_password_hash, check_password_hash
from mtranslate import translate
import re
import pickle
import operator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from scipy.sparse import csr_matrix
from pandas.api.types import is_numeric_dtype
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_mail import Mail, Message
from twilio.rest import Client
import random
import secrets

import warnings

warnings.filterwarnings("ignore")

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

app.config['SECRET_KEY'] = '8BYkEfBA6O6donzWlSihBXox7C0sKR6b'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///master.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Twilio credentials
TWILIO_ACCOUNT_SID = 'AC121bca03050f7749927a491cf533c5bb'
TWILIO_AUTH_TOKEN = 'bad3b3db6f14fefed5a5318c53265192'
TWILIO_PHONE_NUMBER = '+15642167061'

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))
book_data = pd.read_csv('main_dataset.csv')
# Set the environment variable for Tesseract OCR data path
os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata'

# Define language codes for Tesseract and gTTS
LANGUAGE_CODES = {
    "English": {"tesseract": "eng", "gtts": "en"},
    "Portuguese": {"tesseract": "por", "gtts": "pt"},
    "Spanish": {"tesseract": "spa", "gtts": "es"},
    "French": {"tesseract": "fra", "gtts": "fr"},
    "German": {"tesseract": "deu", "gtts": "de"},
    "Hindi": {"tesseract": "hin", "gtts": "hi"},
    "Marathi": {"tesseract": "mar", "gtts": "mr"},
    "Tamil": {"tesseract": "tam", "gtts": "ta"},
    "Telugu": {"tesseract": "tel", "gtts": "te"},
    "Malayalam": {"tesseract": "mal", "gtts": "ml"},
    "Gujarati": {"tesseract": "guj", "gtts": "gu"},
    "Kannada": {"tesseract": "kan", "gtts": "kn"},
    "Bengali": {"tesseract": "ben", "gtts": "bn"},
}

indian_lang = ["hi", "mr", "ta", "te", "ml", "gu", "kn", "bn"]


# Define your User model
class User(db.Model):
    email = db.Column(db.String(120), primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    password = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.Text, nullable=False)


class Audio(db.Model):
    __tablename__ = 'audio'

    filename = db.Column(db.String(255), primary_key=True)
    owner = db.Column(db.String(255), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.now, nullable=False)

    def __repr__(self):
        return f"Audio(filename={self.filename}, owner={self.owner}, datetime_created={self.date_created})"


class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone_number = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    message = db.Column(db.Text, nullable=False)


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.

# Generate OTP
def generate_otp():
    return str(random.randint(1000, 9999))


# Send OTP via SMS
# Send OTP via SMS
def send_otp_sms(mobile_number, otp):
    message = client.messages.create(
        body=f'Your OTP is: {otp}',
        from_=TWILIO_PHONE_NUMBER,
        to=mobile_number
    )
    return message.sid


@app.route('/send-otp', methods=['POST'])
def send_otp():
    mobile_number = "+91" + request.form['phone']
    otp = generate_otp()
    # send_otp_sms(mobile_number, otp)
    print(otp)
    session['otp'] = otp
    return "OTP sent successfully"


@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    user_otp = request.form['otp']
    if user_otp == session.get('otp'):
        return "Verified"
    else:
        return "Wrong OTP"


@app.route('/')
def hello_world():
    if 'email' in session:
        return redirect('/dashboard')
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        phone = "+91" + request.form['phone']
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Check if the email already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return "Email already exists"

        # Encrypt the password
        hashed_password = generate_password_hash(password)

        # Create a new user
        new_user = User(email=email, username=username, password=hashed_password, phone=phone, timestamp=timestamp)
        db.session.add(new_user)
        db.session.commit()

        # Start a session and redirect to dashboard
        session['email'] = email
        return redirect('/dashboard')

    return render_template("signup.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        phone = request.form['phone']

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            return "Invalid email or password"

        session['email'] = email
        return redirect('/dashboard')

    return render_template("login.html")


@app.route('/contact', methods=['POST'])
def contact_form():
    if request.method == 'POST':
        name = request.form['name']
        phone_number = request.form['phone_number']
        email = request.form['email']
        message = request.form['message']

        new_contact = Contact(name=name, phone_number=phone_number, email=email, message=message)
        db.session.add(new_contact)
        db.session.commit()

        return render_template("index.html")


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    return render_template("admin.html")


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Verify credentials
        if email == "admin@gmail.com" and password == "Admin@4327":
            # Credentials match, redirect to admin dashboard
            return redirect(url_for('admin_dashboard'))
        else:
            # Credentials do not match, re-render login page with error message
            error = "Invalid email or password. Please try again."
            return render_template('admin_login.html', error=error)

    # If GET request, render the login page
    return render_template('admin_login.html')


@app.route('/admin/dashboard')
def admin_dashboard():
    # Query the database to get the count of users
    total_users = db.session.query(func.count(User.email)).scalar()

    # Query the database to get the count of users by month
    user_counts = db.session.query(func.strftime('%Y-%m', User.timestamp).label('month'),
                                   func.count(User.email)).group_by('month').all()

    # Extract the labels (months) and user counts from the query results
    labels = [row[0] for row in user_counts]
    users = [row[1] for row in user_counts]

    # Query the database to get the count of audio files by date
    audio_counts = db.session.query(func.strftime('%Y-%m-%d', Audio.date_created).label('date'),
                                    func.count(Audio.filename)).group_by('date').all()

    # Extract the labels (dates) and audio counts from the query results
    audio_labels = [row[0] for row in audio_counts]
    audio_files = [row[1] for row in audio_counts]

    # Query the database to get user details with filenames
    users_with_filenames = db.session.query(User.username, User.email, User.phone, User.timestamp,
                                            func.group_concat(Audio.filename)).join(Audio, User.email == Audio.owner,
                                                                                    isouter=True).group_by(
        User.email).all()

    return render_template('admin_dashboard.html', total_users=total_users, labels=labels, users=users,
                           audio_labels=audio_labels, audio_files=audio_files,
                           users_with_filenames=users_with_filenames)


@app.route('/admin/users')
def admin_users():
    # Query the database to get user details
    users = User.query.all()
    total_users = db.session.query(func.count(User.email)).scalar()
    user_details = [{'email': user.email, 'username': user.username, 'phone': user.phone, 'timestamp': user.timestamp}
                    for user in users]
    print(user_details)
    return render_template('admin_users.html', users=user_details, total_users=total_users)


@app.route('/admin/audiofiles')
def admin_audiofiles():
    # Query the database to get audio file details
    audiofiles = Audio.query.all()
    total_users = db.session.query(func.count(User.email)).scalar()
    return render_template('admin_audiofiles.html', audiofiles=audiofiles, total_users=total_users)


@app.route('/admin/feedbacks')
def admin_feedback():
    # Query the database to get audio file details
    feedbacks = Contact.query.all()
    total_users = db.session.query(func.count(User.email)).scalar()
    return render_template('admin_feedback.html', feedbacks=feedbacks, total_users=total_users)


@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        otp = request.form['otp']
        if otp == session.get('otp'):
            # OTP verification successful, redirect to dashboard
            return redirect('/dashboard')
        else:
            # OTP verification failed, redirect to signup or login
            return redirect('/signup')  # Or return redirect('/login')

    return render_template("verify.html")


@app.route('/dashboard')
def dashboard():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        username = user.username if user else None
        return render_template("dashboard.html", username=username)
    else:
        return redirect('/login')


@app.route('/converttoaudio', methods=['POST', 'GET'])
def convert_to_audio():
    if 'email' not in session:
        return redirect('/login')

    uploaded_file = request.files['uploaded_file']
    voice = request.form['voice']
    speed = int(request.form['speed'])
    target_language = request.form['language']

    if uploaded_file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.filename.endswith(".docx"):
        text = extract_text_from_docx(uploaded_file)
    else:
        return "Unsupported File Type"
    print("Targeted languageL ")
    print(target_language)
    if target_language != 'en' or target_language != 'mr' or target_language != 'hi' or target_language != 'te' or target_language != 'ta' or target_language != 'gu' or target_language != 'ml' or target_language != 'kn' or target_language != 'bn':
        print("In SIMPLE TRANS ==================")
        tr_text = translate_text(text, target_language)

    if not text:
        return "Text extraction failed. Please check the file and try again."

    filename = os.path.join(app.static_folder, 'audiofiles', os.path.splitext(uploaded_file.filename)[0] + '.mp3')
    audiofilename = os.path.splitext(uploaded_file.filename)[0] + '.mp3'
    print(target_language + "-----------------------")
    if LANGUAGE_CODES[target_language]['gtts'] in indian_lang:
        print(LANGUAGE_CODES[target_language]['gtts'])
        language = LANGUAGE_CODES[target_language]['gtts']
        print(text)
        translated_text = translate(text, language)
        convert_text_to_audio_native(translated_text, language, audiofilename, filename)
    else:
        convert_text_to_audio(tr_text, voice, speed, filename, target_language)

    # Save the audio file path to the session
    session['audio_file_path'] = audiofilename

    # Save data to the audio table
    new_audio = Audio(filename=audiofilename, owner=session['email'])
    db.session.add(new_audio)
    db.session.commit()

    print(session['audio_file_path'])
    user = User.query.filter_by(email=session['email']).first()
    username = user.username if user else None
    # Render the audioplayer.html template
    return render_template("audioplayer.html", audiofilename=audiofilename, username=username)


def convert_text_to_audio_native(text, target_language, audiofilename, filename):
    print("IN New Convert")
    tts = gTTS(text=text, lang=target_language)
    tts.save(filename)


def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text


def extract_text_from_docx(file):
    doc_reader = docx.Document(file)
    text = ""
    for paragraph in doc_reader.paragraphs:
        text += paragraph.text + '\n'
    return text


def convert_text_to_audio(text, voice, speed, filename, target_language):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    print(voices)
    if voice == 'male':
        engine.setProperty('voice', voices[0].id)
    else:
        engine.setProperty('voice', voices[1].id)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate * speed / 100)

    gtts_language_code = LANGUAGE_CODES[target_language]['gtts']  # Use target_language here
    engine.setProperty('language', gtts_language_code)

    engine.save_to_file(text, filename)
    engine.runAndWait()


def translate_text(text, target_language):
    translator = Translator()
    print("This is language code ::::")
    print(target_language)
    print(text)
    translated_text = translator.translate(text, src='auto', dest=target_language).text
    print("Translated Text: ", translated_text)
    return translated_text


@app.route('/audiobook')
def audiobook():
    if 'email' in session:

        # Calculate the remaining tries count
        remainingTries = 10 - Audio.query.filter_by(owner=session['email']).count()

        user = User.query.filter_by(email=session['email']).first()
        username = user.username if user else None
        return render_template("audiobook.html", username=username, remainingTries=remainingTries)
    else:
        return redirect('/login')


@app.route('/history')
def history():
    if 'email' in session:
        # Fetch audio files for the current user
        user = User.query.filter_by(email=session['email']).first()
        username = user.username if user else None
        audio_files = Audio.query.filter_by(owner=session['email']).all()
        return render_template("history.html", audio_files=audio_files, username=username)
    else:
        return redirect('/login')


@app.route('/audioplayer/<filename>')
def audioplayer(filename):
    # Construct the path to the audio file
    audio_file_path = os.path.join(app.static_folder, 'audiofiles', filename)

    # Check if the file exists
    if not os.path.exists(audio_file_path):
        return "File not found", 404

    # Render the audioplayer.html template
    user = User.query.filter_by(email=session['email']).first()
    username = user.username if user else None
    return render_template("audioplayer.html", audiofilename=filename, username=username)


@app.route('/recommend')
def recommend_ui():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        username = user.username if user else None
        return render_template('recommend.html', username=username)
    else:
        return redirect('/login')


@app.route('/recommend_books', methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    indices = np.where(pt.index == user_input)[0]
    if len(indices) > 0:
        index = indices[0]
        similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

        data = []
        for i in similar_items:
            item = []
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

            data.append(item)
        return render_template('recommend.html', data=data)
    else:
        error_message = "Book not found. Please try again."
        return render_template('recommend.html', error_message=error_message)


@app.route('/recommendation')
def recommendation():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        username = user.username if user else None
        return render_template('recommendation.html',
                               book_name=list(popular_df['Book-Title'].values),
                               author=list(popular_df['Book-Author'].values),
                               image=list(popular_df['Image-URL-M'].values),
                               votes=list(popular_df['num_ratings'].values),
                               rating=list(popular_df['avg_rating'].values),
                               username=username
                               )
    else:
        return redirect('/login')


@app.route('/recommend_age_page')
def recommend_age_page():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        username = user.username if user else None
        return render_template("recommendation_age_based.html", username=username)
    else:
        return redirect('/login')


@app.route('/recommend_books_age', methods=['POST'])
def recommend_books_age():
    user = User.query.filter_by(email=session['email']).first()
    username = user.username if user else None
    age = int(request.form['age'])
    children_categories = ['Childrens-Books', 'Graphic-Novels-Anime-Manga', 'Teen-Young-Adult', 'Crafts-Hobbies']
    adult_categories = ['Medical', 'Science-Geography', 'Art-Photography', 'Biography', 'Business-Finance-Law',
                        'Computing', 'Crafts-Hobbies', 'Crime-Thriller', 'Dictionaries-Languages', 'Entertainment',
                        'Food-Drink', 'Health', 'History-Archaeology', 'Home-Garden', 'Humour', 'Mind-Body-Spirit',
                        'Natural-History', 'Personal-Development', 'Poetry-Drama', 'Reference', 'Religion', 'Romance',
                        'Science-Fiction-Fantasy-Horror', 'Society-Social-Sciences', 'Sport', 'Technology-Engineering',
                        'Transport', 'Travel-Holiday-Guides']
    senior_categories = ['Medical', 'Entertainment', 'Health', 'Home-Garden', 'Humour', 'Mind-Body-Spirit', 'Religion',
                         'Travel-Holiday-Guides']
    if age <= 18:
        category = children_categories
    elif age <= 40:
        category = adult_categories
    else:
        category = senior_categories

    # Filter the dataset for the selected category and get the top-rated books
    top_books = book_data[book_data['category'].isin(category)].nlargest(50, 'book_depository_stars')
    top_books = top_books.sample(frac=1).reset_index(drop=True)

    # top_books['image'] = top_books['isbn'].apply(lambda x: f'https://covers.openlibrary.org/b/isbn/{x}-L.jpg')

    recommended_books_json = top_books.to_json(orient='records')
    return render_template("recommendation_age_based.html", username=username, data=json.loads(recommended_books_json))


@app.route('/delete_audio', methods=['POST'])
def delete_audio():
    filename = request.form.get('filename')

    # Delete the audio file from the 'audiofiles' folder
    audio_path = os.path.join(app.static_folder, 'audiofiles', filename)
    os.remove(audio_path)

    # Delete the entry from the database
    # Assuming 'audio' is your SQLAlchemy model for the audio files
    audio_entry = Audio.query.filter_by(filename=filename).first()
    db.session.delete(audio_entry)
    db.session.commit()

    return 'Audio file deleted successfully'


@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/')


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
