from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

# Define the URL and User-Agent header
url = 'https://www.zdnet.com/article/how-to-use-chatgpt-to-summarize-a-book-article-or-research-paper/'
headers = {'User-Agent': 'Mozilla/5.0'}

# Create a request object with headers
req = urllib.request.Request(url, headers=headers)

    # Open the URL with the request object
with urllib.request.urlopen(req) as response:
    html = response.read()

    # Process the HTML content
print(text_from_html(html))

#html = urllib.request.urlopen('http://www.nytimes.com/2009/12/21/us/21storm.html', headers={'User-Agent': 'Mozilla/5.0'}).read()
#print(text_from_html(html))