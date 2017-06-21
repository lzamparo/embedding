import concurrent.futures
import urllib.request
import time

URLS = ['http://www.foxnews.com/',
        'http://www.cnn.com/',
        'http://europe.wsj.com/',
        'http://www.bbc.co.uk/',
        'http://some-made-up-domain.com/',
        'http://www.nytimes.com',
        'http://www.facebook.com',
        'http://www.silversevensens.com',
        'http://www.wakingthered.com',
        'http://www.twitter.com',
        'http://www.google.com',
        'http://www.economist.com',
        'http://www.cbc.ca',
        'http://www.newyorker.com',
        'http://www.nyc.gov']

# Retrieve a single page and report the url and contents
def load_url(url, timeout):
    conn = urllib.request.urlopen(url, timeout=timeout)
    return conn.read()

workers=5
t0 = time.time()

# We can use a with statement to ensure threads are cleaned up promptly
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    # Start the load operations and mark each future with its URL
    future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (url, exc))
        else:
            print('%r page is %d bytes' % (url, len(data)))
t1 = time.time()
print("All URL ops took ", t1 - t0, " seconds with ", workers, " workers")

# Serial version
my_URLs = []
t2 = time.time()
for url in URLS:
    try:
        my_URLs.append(load_url(url, 60))
    except:
        continue
    
for url, res in zip(URLS, my_URLs):
    try:
        print(url, " is ", len(res), " bytes ")
    except Exception as exc:
        print(url, " messed up ", exc)
t3 = time.time()
print("All URL ops took ", t3 - t2, " seconds serially")
    