import urllib.request,urllib.error,http.cookiejar,json,re
from acquire import PUBLIC
jar=http.cookiejar.CookieJar()
opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
opener.addheaders=[('User-Agent','Mozilla/5.0')]
with opener.open(PUBLIC+'/file/1280808267878',timeout=45) as r:
    html=r.read().decode(); final=r.geturl()
print('final_public_page',final)
match=re.search(r'Box\.config\s*=\s*',html)
config=json.JSONDecoder().raw_decode(html[match.end():])[0]
payload=json.dumps({'fileIDs':['1280808267878']}).encode()
req=urllib.request.Request('https://rochester.app.box.com/app-api/enduserapp/elements/tokens',data=payload,
    headers={'Content-Type':'application/json','X-Request-Token':config['requestToken'],'Referer':PUBLIC+'/file/1280808267878'})
try:
    with opener.open(req,timeout=45) as r:
        response=json.load(r)
        print('token_request_status',r.status,'response_keys',list(response))
        for k,v in response.items():
            if isinstance(v,dict): print('nested_keys',list(v))
        m=re.search(r'Box\.prefetchedData\s*=\s*',html)
        meta=json.JSONDecoder().raw_decode(html[m.end():])[0]['preview_metadata']
        print('download_path',urllib.parse.urlsplit(meta['authenticated_download_url']).path,'file_shared_link',meta['shared_link'])
        req=urllib.request.Request(meta['authenticated_download_url'],headers={
            'Authorization':'Bearer '+response['1280808267878']['read'],
            'BoxApi':'shared_link='+final.split('/file/')[0],'Referer':final})
        with opener.open(req,timeout=45) as f:
            print('download_status',f.status,'content_type',f.headers.get('Content-Type'),'first_bytes_count',len(f.read(100)))
except urllib.error.HTTPError as e:
    print('HTTP_status',e.code,'body_bytes',len(e.read()))
legacy='https://rochester.app.box.com/index.php?rm=box_download_shared_file&shared_name=AtaxiaDataset&file_id=1280808267878'
try:
    with opener.open(legacy,timeout=45) as f:
        print('legacy_status',f.status,'content_type',f.headers.get('Content-Type'),'first_bytes_count',len(f.read(100)))
except urllib.error.HTTPError as e: print('legacy_HTTP_status',e.code)
