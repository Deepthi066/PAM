from twilio.twiml.voice_response import VoiceResponse
from flask import Flask, Response

app = Flask(__name__)

@app.route('/twiml', methods=['GET'])
def generate_twiml():
    response = VoiceResponse()
    response.say('')
    return Response(str(response), mimetype='text/xml')

if __name__ == '__main__':
    app.run(debug=True)
