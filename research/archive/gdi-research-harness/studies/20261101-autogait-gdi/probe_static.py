import urllib.request,re
for path in ['app.33b2d40aba.js','main.566912f832.js','vendors~main.ad8e739bd3.js']:
    request=urllib.request.Request('https://cdn01.boxcdn.net/enduser/'+path,headers={'User-Agent':'Mozilla/5.0','Referer':'https://rochester.app.box.com/'})
    with urllib.request.urlopen(request, timeout=45) as r:
        s=r.read().decode()
    for pattern in ['elements/tokens','box_download_shared_file','download_shared','fileIDs','sharedName']:
        hits=list(re.finditer(re.escape(pattern),s))
        print(path,pattern,len(hits))
        for m in hits[:3]:print(s[max(0,m.start()-250):m.end()+300])
