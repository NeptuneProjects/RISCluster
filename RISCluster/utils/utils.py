from email.message import EmailMessage
import os
import smtplib
import ssl

from dotenv import load_dotenv
from twilio.rest import Client

def notify(msgsubj, msgcontent):
    '''Written by William Jenkins, 19 June 2020, wjenkins@ucsd.edu3456789012
    Scripps Institution of Oceanography, UC San Diego
    This function uses the SMTP and Twilio APIs to send an email and WhatsApp
    message to a user defined in environmental variables stored in a .env file
    within the same directory as this module.  Sender credentials are stored
    similarly.'''
    load_dotenv()
    msg = EmailMessage()
    msg['Subject'] = msgsubj
    msg.set_content(msgcontent)
    username = os.getenv('ORIG_USERNAME')
    password = os.getenv('ORIG_PWD')
    try:
        # Create a secure SSL context
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com',
                              port=465, context=context) as s:
            s.login(username, password)
            receiver_email = os.getenv('RX_EMAIL')
            s.sendmail(username, receiver_email, msg.as_string())
            print('Job completion notification sent by email.')
    except:
        print('Unable to send email notification upon job completion.')
        pass
    try:
        client = Client()
        orig_whatsapp_number = 'whatsapp:' + os.getenv('ORIG_PHONE_NUMBER')
        rx_whatsapp_number = 'whatsapp:' + os.getenv('RX_PHONE_NUMBER')
        msgcontent = '*' + msgsubj + '*\n' + msgcontent
        client.messages.create(body=msgcontent,
                               from_=orig_whatsapp_number,
                               to=rx_whatsapp_number)
        print('Job completion notification sent by WhatsApp.')
    except:
        print('Unable to send WhatsApp notification upon job completion.')
        pass
