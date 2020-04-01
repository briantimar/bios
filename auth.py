import json
import tweepy
import os

ROOT=os.path.dirname(os.path.realpath(__file__))

KEYS = os.path.join(ROOT, "keys.json")
TOKENS = os.path.join(ROOT, "tokens.json")

def get_keys():
    try:    
        with open(KEYS) as f:
            return json.load(f)
    except FileNotFoundError:
        key = os.environ.get("CONSUMER_KEY")
        secret = os.environ.get("CONSUMER_SECRET")
        return {'consumer_key': key, 'consumer_secret': secret}

def get_tokens():
    with open(TOKENS) as f:
        return json.load(f)

def get_app_only_auth():
    """ Returns authentication object for application only (no user credentials required) """
    #application keys
    keys = get_keys()
    return tweepy.AppAuthHandler(keys['consumer_key'], keys['consumer_secret'])

def get_root_user_auth():
    """ Returns authentication for the root user (this allows for full access to their twitter profile)"""
    keys = get_keys()
    tokens = get_tokens()
    auth = tweepy.OAuthHandler(keys['consumer_key'], keys['consumer_secret'])
    auth.set_access_token(tokens['access_token'], tokens['access_token_secret'])
    return auth

