import requests
from bs4 import BeautifulSoup
import json
import time
from string import punctuation
import re
from collections import defaultdict
import numpy as np
import pandas as pd


def get_reddit():
    url = 'https://old.reddit.com/r/AmItheAsshole/controversial/'
    params = {}
    data = open_data()
    for i in range(100):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36'
        }
        response = requests.get(url, headers=headers, params=params)
        print(response.status_code)
        if response.status_code != 200:
            print('Failed to fetch page')
            break
        soup = BeautifulSoup(response.text, 'html.parser')
        # find class of top-matter
        posts = soup.find_all('div', class_='top-matter')
        for post in posts:
            title_card = post.find('p', class_='title')
            title = title_card.find('a', class_='title may-blank').text
            print(title)
            # check if prefix is 'AITA'
            if title.startswith('AITA'):
                parsed = parse_data(title_card)
                if parsed is not None:
                    data[title] = parsed
        # get next page
        next_button = soup.find('span', class_='next-button')
        if next_button is None:
            break
        else:
            next_page = next_button.find('a')
            if next_page is None:
                break
            else:
                url = next_page['href']
    save_data(data)


def open_data():
    with open('./data/reddit/data.json', 'r') as f:
        data = json.load(f)
    return data


def save_data(data):
    with open('./data/reddit/data.json', 'w') as f:
        json.dump(data, f, indent=2)


def get_p():
    data = open_data()
    length = len(data)
    for i, key in enumerate(data):
        print(f'{i}/{length}')
        post = data[key]
        href = post['href']
        if post.get('p') is None:
            p = get_post(href)
            if p is not None:
                post['p'] = p
                data[key] = post
            # save every 10 posts
            if i % 10 == 0:
                save_data(data)
    save_data(data)


def get_post(href: str) -> str | None:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36'
        }
        base_url = 'https://old.reddit.com'
        response = back_off(base_url + href, headers)
        if response is None:
            return None
        soup = BeautifulSoup(response.text, 'html.parser')
        body = soup.find('div', id='siteTable')
        post = body.find('div', class_='md')
        paragraphs = post.find_all('p')
        text = ''
        for p in paragraphs:
            text += p.text + ' '
        return text
    except Exception as e:
        print(e)
        return None


def create_vocab():
    data = open_data()
    vocab = defaultdict(int)

    # if word is in 96% of posts, remove it
    cutoff_upper = len(data) * 0.99

    # if word is in 4% of posts, remove it
    cutoff_lower = len(data) * 0.005

    for key in data:
        post = data[key]
        p = post['p']
        for punct in punctuation:
            if punct in p:
                vocab[punct] += 1

        # remove punc
        pattern = f"[{re.escape(punctuation)}]"
        p = re.sub(pattern, '', p)
        p = p.lower()
        p = p.replace('"', ' ')
        p = p.replace('\n', ' ')
        p = p.replace('.', ' ')
        words = p.split(' ')

        seen = set()
        for word in words:
            if word in seen:
                continue
            vocab[word] += 1
            seen.add(word)

    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    for word, count in sorted_vocab:
        if count > cutoff_upper:
            del vocab[word]
        if count < cutoff_lower:
            del vocab[word]

    print(len(vocab.keys()))
    with open('./data/reddit/vocab.json', 'w') as f:
        json.dump(list(vocab.keys()), f, indent=2)


def back_off(url: str, headers: dict) -> requests.Response | None:
    back_off = 1
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response
        if response.status_code == 429:
            print(f'Rate limited, backing off for {back_off} seconds')
            time.sleep(back_off)
            back_off *= 2
        else:
            return None


def parse_data(title_card) -> dict | None:
    title = title_card.find('a', class_='title may-blank')
    href = title['href']
    ahole = title_card.find('span', class_='linkflairlabel')
    is_asshole = False
    if ahole is None:
        return None
    if ahole.text.startswith('Not'):
        is_asshole = False
    elif ahole.text.startswith('Asshole'):
        is_asshole = True
    else:
        return None
    return {
        'title': title.text,
        'href': href,
        'is_asshole': int(is_asshole)
    }


def count():
    data = open_data()
    asshole = 0
    not_asshole = 0
    for key in data:
        if data[key]['is_asshole']:
            asshole += 1
        else:
            not_asshole += 1
    print(f'Asshole: {asshole}')
    print(f'Not Asshole: {not_asshole}')


def open_vocab():
    with open('./data/reddit/vocab.json', 'r') as f:
        vocab = json.load(f)
    return vocab


def load_reddit_data():
    from utils.utils import one_hot
    data = open_data()
    vocab = open_vocab()
    x_list = []
    y_list = []
    for post in data.values():
        x = np.zeros(len(vocab))
        p = post['p']
        p = p.lower()
        p = p.replace('"', ' ')
        p = p.replace('\n', ' ')
        p = p.replace('.', ' ')
        words = p.split(' ')
        for word in words:
            if word in vocab:
                x[vocab.index(word)] += 1
        x_list.append(x)
        y_list.append(post['is_asshole'])

    # remove 400 non-asshole posts
    new_x = []
    new_y = []

    print(len(x_list))
    removed = 0
    i = 0
    while i < len(x_list):
        y = y_list[i]
        if y == 0 and removed < 400:
            removed += 1
        else:
            new_x.append(x_list[i])
            new_y.append(y_list[i])
        i += 1
    print(len(new_x))

    # create containing x column and y column
    df = pd.DataFrame({'x': new_x, 'y': new_y})

    # shuffle
    df = df.sample(frac=1)

    x = np.array(df['x'].to_list())
    y = np.array(df['y'].to_list())

    print(x.shape)
    y = one_hot(y).T
    print(x.shape)
    print(y.shape)
    print(x[0])
    print(y[0])

    x_train = x[:-200]
    y_train = y[:-200]
    x_test = x[-200:]
    y_test = y[-200:]

    return x_test, y_test, x_train, y_train
