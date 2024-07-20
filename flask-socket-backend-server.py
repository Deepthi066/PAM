from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from flask_cors import CORS
import time
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

cors = CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", cors_credentials=True)

# Function to send random numbers to the client
def send_random_numbers():
    while True:
        number1 = random.randint(0, 100)
        socketio.emit('random_number', {'message':number1}, namespace='/test')
        time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')


# SocketIO event to handle client connection
@socketio.on('connect', namespace='/test')
def handle_connect():
    print('Client connected')

# SocketIO event to handle client disconnection
@socketio.on('disconnect', namespace='/test')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.start_background_task(send_random_numbers)
    socketio.run(app)
