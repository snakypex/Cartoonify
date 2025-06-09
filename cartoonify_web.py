import os
import sqlite3
import time
import uuid
from functools import wraps
from tempfile import NamedTemporaryFile
from threading import Thread, Lock
from flask import Flask, request, send_file, redirect, session, url_for
from cartoonify_video import process_video
import stripe
import paypalrestsdk
from flask_dance.contrib.google import make_google_blueprint, google

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret')

google_bp = make_google_blueprint(
    client_id=os.getenv('GOOGLE_OAUTH_CLIENT_ID', ''),
    client_secret=os.getenv('GOOGLE_OAUTH_CLIENT_SECRET', ''),
    scope=['profile', 'email'],
    redirect_to='index'
)
app.register_blueprint(google_bp, url_prefix='/login')

# Database utilities
DB_PATH = os.getenv('PAYMENT_DB_PATH', 'payments.db')


def init_db():
    """Create the users table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, expires INTEGER)"
        )
        conn.commit()


def set_subscription_end(email, expires_at):
    """Store subscription expiration timestamp."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO users (email, expires) VALUES (?, ?)",
            (email, int(expires_at)),
        )
        conn.commit()


def subscription_expires(email):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT expires FROM users WHERE email=?", (email,)
        ).fetchone()
        return int(row[0]) if row else 0


def extend_subscription(email, months=1):
    now = int(time.time())
    current = subscription_expires(email)
    if current > now:
        expires = current + months * 30 * 24 * 3600
    else:
        expires = now + months * 30 * 24 * 3600
    set_subscription_end(email, expires)


def is_subscribed(email):
    """Check if the user's subscription is active."""
    return subscription_expires(email) > time.time()


init_db()


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not google.authorized:
            return redirect(url_for('google.login'))
        # Store user email in session for later use
        if 'email' not in session:
            resp = google.get('/oauth2/v2/userinfo')
            if resp.ok:
                session['email'] = resp.json().get('email')
        return f(*args, **kwargs)
    return wrapper


def subscription_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        email = session.get('email')
        if not email or not is_subscribed(email):
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return wrapper

stripe.api_key = os.getenv('STRIPE_SECRET_KEY', '')
paypalrestsdk.configure({
    'mode': 'sandbox',
    'client_id': os.getenv('PAYPAL_CLIENT_ID', ''),
    'client_secret': os.getenv('PAYPAL_CLIENT_SECRET', '')
})

progress_jobs = {}
job_queue = []
queue_lock = Lock()

def queue_worker():
    while True:
        with queue_lock:
            if job_queue:
                job_id = job_queue.pop(0)
            else:
                job_id = None
        if job_id is None:
            time.sleep(1)
            continue

        def cb(p):
            progress_jobs[job_id]['progress'] = int(p * 100)

        in_path = progress_jobs[job_id]['input']
        out_path = progress_jobs[job_id]['output']
        process_video(in_path, out_path, progress_callback=cb)
        progress_jobs[job_id]['progress'] = 100
        os.unlink(in_path)

Thread(target=queue_worker, daemon=True).start()

LOGIN_HTML = """<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Login</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; background: #f4f4f4; padding: 20px; border-radius: 8px; text-align: center; }
    .button { display: inline-block; padding: 10px 20px; margin: 10px; background: #007bff; color: #fff; text-decoration: none; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>Sign in</h1>
  <a class='button' href='/login/google'>Login with Google</a>
</body>
</html>
"""

INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Cartoonify SaaS</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; background: #f4f4f4; padding: 20px; border-radius: 8px; }
    h1, h2 { text-align: center; }
    .button { display: inline-block; padding: 10px 20px; margin: 10px 5px; background: #007bff; color: #fff; text-decoration: none; border-radius: 4px; }
    form { margin-top: 20px; text-align: center; }
    input[type=file] { margin-bottom: 10px; }
  </style>
</head>
<body>
  <h1>Cartoonify your videos</h1>
  <p style='text-align:center;'>Welcome {email}</p>
  {subscribe_section}
  {upload_form}
</body>
</html>
"""

@app.route('/')
def index():
    if not google.authorized:
        return LOGIN_HTML
    # Retrieve user email
    if 'email' not in session:
        resp = google.get('/oauth2/v2/userinfo')
        if resp.ok:
            session['email'] = resp.json().get('email', '')
    email = session.get('email', '')

    if is_subscribed(email):
        exp = subscription_expires(email)
        date_str = time.strftime('%Y-%m-%d', time.localtime(exp))
        subscribe_section = f"<p style='text-align:center;'>Subscription active until {date_str}.</p>"
        upload_form = (
            "<h2>Upload a video</h2>"
            "<form method='post' enctype='multipart/form-data' action='/upload'>"
            "  <input type='file' name='video' required><br>"
            "  <input class='button' type='submit' value='Cartoonify'>"
            "</form>"
        )
    else:
        subscribe_section = (
            "<p style='text-align:center;'>Subscribe for only 3.99€ per month to use the service.</p>"
            "<div style='text-align:center;'>"
            "  <a class='button' href='/pay/stripe'>Pay with Stripe</a>"
            "  <a class='button' href='/pay/paypal'>Pay with PayPal</a>"
            "</div>"
        )
        upload_form = ""
    return INDEX_HTML.format(email=email, subscribe_section=subscribe_section, upload_form=upload_form)

@app.route('/upload', methods=['POST'])
@login_required
@subscription_required
def upload():
    file = request.files.get('video')
    if not file:
        return 'No video uploaded', 400

    with NamedTemporaryFile(delete=False, suffix='.mp4') as in_temp:
        file.save(in_temp.name)
        in_path = in_temp.name

    with NamedTemporaryFile(delete=False, suffix='.mp4') as out_temp:
        out_path = out_temp.name

    job_id = uuid.uuid4().hex
    progress_jobs[job_id] = {
        'progress': 0,
        'output': out_path,
        'input': in_path,
    }
    with queue_lock:
        job_queue.append(job_id)

    return redirect(url_for('job_status', job_id=job_id))


@app.route('/job/<job_id>')
@login_required
def job_status(job_id):
    job = progress_jobs.get(job_id)
    if not job:
        return 'Job not found', 404
    with queue_lock:
        if job_id in job_queue:
            position = job_queue.index(job_id) + 1
        else:
            position = 0
    if job['progress'] >= 100 and os.path.exists(job['output']):
        path = job['output']
        del progress_jobs[job_id]
        return send_file(path, as_attachment=True, download_name='cartoonified.mp4', mimetype='video/mp4')
    progress = job['progress']
    if position > 0:
        return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Queued</title></head>
<body>
<h1>Waiting in queue</h1>
<p>Position: {position}</p>
<script>setTimeout(function(){{window.location.reload();}}, 1000);</script>
</body></html>"""
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Processing</title></head>
<body>
<h1>Processing...</h1>
<progress value='{progress}' max='100'></progress>
<p>{progress}%</p>
<script>setTimeout(function(){{window.location.reload();}}, 1000);</script>
</body></html>"""


@app.route('/pay/stripe')
@login_required
def pay_stripe():
    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[{
            'price_data': {
                'currency': 'eur',
                'unit_amount': 399,
                'recurring': {'interval': 'month'},
                'product_data': {'name': 'Cartoonify SaaS monthly'},
            },
            'quantity': 1,
        }],
        mode='subscription',
        success_url=url_for('payment_success', _external=True),
        cancel_url=url_for('index', _external=True),
    )
    return redirect(session.url)


@app.route('/pay/paypal')
@login_required
def pay_paypal():
    payment = paypalrestsdk.Payment({
        'intent': 'sale',
        'payer': {'payment_method': 'paypal'},
        'redirect_urls': {
            'return_url': url_for('payment_success', _external=True),
            'cancel_url': url_for('index', _external=True),
        },
        'transactions': [{
            'amount': {
                'total': '3.99',
                'currency': 'EUR'
            },
            'description': '1 month Cartoonify SaaS subscription'
        }]
    })
    if payment.create():
        for link in payment.links:
            if link.rel == 'approval_url':
                return redirect(link.href)
    return 'Error creating PayPal payment', 500


@app.route('/payment/success')
@login_required
def payment_success():
    email = session.get('email')
    if email:
        extend_subscription(email, 1)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
