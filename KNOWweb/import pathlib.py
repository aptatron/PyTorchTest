import pathlib
from pathlib import Path
import requests

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) '}

url = 'https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py'

with open('helper_functions.py', 'wb') as f:
    f.write(requests.get(url, headers=headers).content)

