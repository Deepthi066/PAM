from twilio.rest import Client
from dotenv import load_dotenv


account_sid = load_dotenv("ACCOUNT_SID")
auth_token = load_dotenv("AUTH_TOKEN")
client = Client(account_sid, auth_token)


def send_sms_notification(body):
  message = client.messages.create(
      body=body,
    frm_="+15854604752",
    to="+919513881008"
  )

  print(message.sid)