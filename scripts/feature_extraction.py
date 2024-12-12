from urllib.parse import urlparse

def extract_features_from_url(url):
    parsed_url = urlparse(url)
    features = [
        url.count('.'), len(parsed_url.netloc.split('.')) - 2, url.count('/') - 2, len(url), url.count('-'),
        parsed_url.netloc.count('-'), int('@' in url), int('~' in url), url.count('_'),
        url.count('%'), len(parsed_url.query.split('&')) if parsed_url.query else 0,
        sum(c.isdigit() for c in url), sum(c.isalpha() for c in url),
        parsed_url.path.count('/'), url.lower().count('http') - 1
    ]
    return features
