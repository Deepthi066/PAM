import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Email configuration
sender_email = ""
receiver_email = ""  
password = ""

# Create a multipart message
message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = ""

# Add body to email
body = ""
message.attach(MIMEText(body, "plain"))

try:
    # Connect to the SMTP server
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    
    # Log in to the SMTP server
    server.login(sender_email, password)
    
    # Send email
    server.sendmail(sender_email, receiver_email, message.as_string())
    
    print("Email sent successfully to yourself!")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Quit the SMTP server
    server.quit()
