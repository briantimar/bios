import tweepy
from tweepy.error import RateLimitError
from auth import get_root_user_auth
import time
from collections import deque
import pickledb
import sys

DBPATH = "bios.json"

def catch_rate_limit(f):
    """Catch rate limit error into code."""
    def caught(*args, **kwargs):
        try:
            return f(*args, **kwargs), True
        except RateLimitError:
            return None, False
    return caught

def retry(f, init_delay=10, max_iter=10):
    """Retry a rate-limited request until success.
        init_delay = initial time between requests; doubles"""
    def repeated(*args, **kwargs):
        resp, success = f(*args, **kwargs)
        ct = 0
        delay = init_delay
        while not success:
            if ct == max_iter:
                raise ValueError("request failed")
            print(f"Rate limit; waiting for {delay} sec", file=sys.stderr, flush=True)
            time.sleep(delay)
            delay *= 2
            ct += 1
            resp, success = f(*args, **kwargs)
        return resp
    return repeated

def retry_rate_limited(f):
    return retry(catch_rate_limit(f), init_delay=10, max_iter=10)

@retry_rate_limited
def get_friends(api, id):
    """Returns friends of the given id."""
    return list(tweepy.Cursor(api.friends_ids, id=id,count=200).items(600))
    
@retry_rate_limited
def get_name_and_bio(api, id):
    """Returns screen name and bio for the given id."""
    u = api.get_user(id)
    return u.screen_name, u.description

def collect_bios(api, start_id, db):
    """store name,bio pairs"""
    user_ids = deque(maxlen=100000)
    seen = set()
    
    def save(name, bio):
        db.set(name, bio)
        
    (name, bio) = get_name_and_bio(api, start_id)
    print(name, bio, flush=True)

    save(name, bio)
    seen.add(start_id)
    user_ids.append(start_id)

    while True:
        if len(user_ids) == 0:
            print("user queue exhausted", flush=True)
            break
        user_id = user_ids.popleft()
        try:
            friend_ids = get_friends(api, user_id)
            for f_id in friend_ids:
                if f_id not in seen:
                    name, bio= get_name_and_bio(api, f_id)
                    save(name, bio)
                    seen.add(f_id)
                    user_ids.append(f_id)
                    print(f"fetched {name}", flush=True)
            db.dump()
            print(f"entries: {db.totalkeys()}")

        except tweepy.error.TweepError as e:
            print(f"failed with {user_id}, tweep error {e.reason} ({e.api_code})", file=sys.stderr, flush=True)

if __name__ == "__main__":
    api = tweepy.API(get_root_user_auth())
    db = pickledb.load(DBPATH, auto_dump=False)
    start_id = "mgboz"
    fname = "bios.txt"
    print(f"Found {db.totalkeys()} entries")
    collect_bios(api, start_id, db)