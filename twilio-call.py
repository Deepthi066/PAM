from twilio.rest import Client

# Your Twilio Account SID and Auth Token
account_sid = ""
auth_token = ""

# Create Twilio client
client = Client(account_sid, auth_token)

# Make a phone call
call = client.calls.create(
    to="",  # Replace with the recipient's phone number
    from_="",  # Your Twilio phone number
    url=""    #-> Twilio Console -> twiML Bins
    #url=''  # URL to TwiML with instructions for the call
)

print(call.sid)  # Print the unique ID of the call
