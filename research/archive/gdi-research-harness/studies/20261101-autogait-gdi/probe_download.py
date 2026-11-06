from acquire import metadata, PUBLIC
import re,json
import urllib.request, urllib.error, urllib.parse
m = metadata(1280808267878)
print('metadata_keys', list(m))
url = m['authenticated_download_url']
print('url_host', urllib.parse.urlsplit(url).netloc)
r = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0','Referer':PUBLIC})
try:
    with urllib.request.urlopen(r, timeout=45) as a:
        print(a.status, a.headers.get('Content-Type'))
except urllib.error.HTTPError as e:
    print('HTTP_status', e.code, 'header_names', list(e.headers), 'body_bytes', len(e.read()))
with urllib.request.urlopen(urllib.request.Request(PUBLIC+'/file/1280808267878', headers={'User-Agent':'Mozilla/5.0'})) as r:
    html=r.read().decode()
for label in ['prefetchedData','config']:
    match=re.search(r'Box\.'+label+r'\s*=\s*',html)
    if match:
        try:
            data=json.JSONDecoder().raw_decode(html[match.end():])[0]
            print(label,'keys',list(data))
            for key,val in data.items():
                if isinstance(val,dict): print(key,'nestedkeys',list(val))
        except ValueError: print(label,'not JSON')
print('token-related identifier names', sorted(set(re.findall(r'[A-Za-z_]*(?:[Tt]oken|[Aa]ccess)[A-Za-z_]*',html))))
print('script resources',[s for s in re.findall(r'<script[^>]+src="([^"]+)"',html) if 'box' in s])
