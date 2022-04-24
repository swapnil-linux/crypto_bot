import os
import json
import requests
import sys
from py3cw.request import Py3CW

TELE_TOKEN = os.environ['TELE_TOKEN']
URL = "https://api.telegram.org/bot{}/".format(TELE_TOKEN)

def getuid(username):
    accountid = os.environ[username]
    return accountid

p3cw = Py3CW(
    key=os.environ['TC_KEY'],
    secret=os.environ['TC_SECRET'],
    request_options={
        'request_timeout': 10,
        'nr_of_retries': 1,
        'retry_status_codes': [502],
        'retry_backoff_factor': 0.1
    }
)

def send_message(text, chat_id):
    url = URL + "sendMessage?text={}&chat_id={}".format(text, chat_id)
    requests.get(url)

def start(arguments, chat_id):
    """Send a message when the command /start is issued."""
    message = "Hello " + arguments
    send_message(message, chat_id)
    send_message('/activedeals - Current Active Deals\n/closeddeals - Last 5 Closed Deals\n/overallstats - Overall Stats',chat_id)

def overallstats(accountid, chat_id):

    error, bots_data = p3cw.request(
        entity = 'bots',
        action = 'stats',
        payload={
        'account_id': accountid,
        }
    )

    profits = bots_data['profits_in_usd']
    overall_gains = bots_data['overall_stats']
    todays_gain = bots_data['today_stats']

    message = "---------------------------------------"
    message += "\nOverall Stats in USD for " + accountid
    message += "\n--------------------------------------"
    for key, value in profits.items():
        amount = "${:,.2f}".format(value)
        message += "\n" + key + ": " + amount

    message += "\n\n-------------------------------"
    message += "\nOverall Gains"
    message += "\n-------------------------------"
    for key, value in overall_gains.items():
        #amount = "${:,.2f}".format(value)
        message += "\n" + key + ": " + value

    message += "\n\n-------------------------------"
    message += "\nTodays Gain"
    message += "\n-------------------------------"
    for key, value in todays_gain.items():
        #amount = "${:,.2f}".format(value)
        message += "\n" + key + ": " + value

    send_message(message, chat_id)

def closeddeals(accountid, chat_id):
    error, deals_data = p3cw.request(
        entity = 'deals',
        payload={
        'account_id': accountid,
        'scope': 'completed',
        'limit': 5
        }
    )
    message = "----------------------------------------"
    message += "\nLast 5 Closed Deals for " + accountid
    message += "\n----------------------------------------"
    for datapoint_deals in deals_data:
        pair = str(datapoint_deals['pair'])
        amount = str(round(float(datapoint_deals['bought_volume']),2))
        profit_percentage = str(datapoint_deals['final_profit_percentage'])
        profit = str(round(float(datapoint_deals['usd_final_profit']),2))
        closedat = str(datapoint_deals['closed_at'])
        message += "\n - " + pair + "($"+ amount +")" + ": Profit:$" + profit + "(" + profit_percentage + "%) Closed At: " + closedat
    send_message(message, chat_id)

def activedeals(accountid, chat_id):
    error, deals_data = p3cw.request(
        entity = 'deals',
        payload={
        'account_id': accountid,
        'scope': 'active',
        }
    )
    message = "----------------------------------------"
    message += "\nCurrent Active Deals for " + accountid
    message += "\n----------------------------------------"
    for datapoint_deals in deals_data:
        pair = str(datapoint_deals['pair'])
        amount = str(round(float(datapoint_deals['bought_volume']),2))
        profit_percentage = str(datapoint_deals['actual_profit_percentage'])
        profit = str(round(float(datapoint_deals['actual_usd_profit']),2))
        created = str(datapoint_deals['created_at'])
        message += "\n - " + pair + "($"+ amount +")" + ": Profit:$" + profit + "(" + profit_percentage + "%)"
    send_message(message, chat_id)

def lambda_handler(event, context):
    chat_id = event['message']['chat']['id']
    fname = event['message']['chat']['first_name']
    username = event['message']['chat']['username']
    accountid = getuid(username)
    command_arguments = event['message']['text'].split()
    command = command_arguments[0]
    arguments = command_arguments[1:]
    if command == "/start":
        if accountid != "invalid":
            start(fname, chat_id)
        else:
            send_message('Hi ' + str(user.first_name) + ' you are not authorised to use this service',chat_id)
    elif command == "/overallstats":
            overallstats(accountid,chat_id)
    elif command == "/closeddeals":
            closeddeals(accountid,chat_id)
    elif command == "/activedeals":
            activedeals(accountid,chat_id)
    else:
        send_message("Command not support", chat_id)

    return {
        'statusCode': 200,
    }
