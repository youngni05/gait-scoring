// Local Wi-Fi preview: only built assets and the public CA certificate are served.
import https from 'node:https';
import http from 'node:http';
import { readFileSync, statSync, createReadStream } from 'node:fs';
import { resolve, extname, sep } from 'node:path';

const host = process.env.PHONE_HOST;
if (!host || !/^\d{1,3}(\.\d{1,3}){3}$/.test(host)) throw new Error('PHONE_HOST must be a local IPv4 address');
const apiUrl = process.env.PHONE_API_URL;
if (!apiUrl || !/^http:\/\/127\.0\.0\.1:\d+$/.test(apiUrl)) throw new Error('PHONE_API_URL must be the local Compose API URL');
const api = new URL(apiUrl);
const root = resolve('mobile/local-output/phone');
const certificates = resolve('local-test-data/phone-tls');
const mime = {'.html':'text/html; charset=utf-8','.js':'text/javascript','.css':'text/css','.json':'application/json','.wasm':'application/wasm','.task':'application/octet-stream','.png':'image/png','.svg':'image/svg+xml','.woff2':'font/woff2'};
const server = https.createServer({key:readFileSync(resolve(certificates,'server.key')),cert:readFileSync(resolve(certificates,'server.crt'))}, (req,res) => {
  const url = new URL(req.url, 'https://localhost');
  if (url.pathname.startsWith('/api/')) {
    const headers = {...req.headers, host:api.host};
    delete headers.origin; delete headers.referer;
    const upstream = http.request({hostname:api.hostname,port:api.port,path:url.pathname.slice(4)+url.search,method:req.method,headers}, response => {
      res.writeHead(response.statusCode, response.headers); response.pipe(res);
    });
    upstream.on('error', () => { if (!res.headersSent) res.writeHead(502); res.end('Local API unavailable'); });
    req.on('aborted',()=>upstream.destroy()); req.pipe(upstream); return;
  }
  if (!['GET','HEAD'].includes(req.method)) { res.writeHead(405); res.end(); return; }
  try {
    const pathname = decodeURIComponent(url.pathname);
    const file = resolve(root, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (!file.startsWith(root + sep) || !statSync(file).isFile()) throw new Error('Not found');
    res.writeHead(200, {'Content-Type':mime[extname(file)] || 'application/octet-stream','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});
    if (req.method === 'HEAD') res.end(); else createReadStream(file).pipe(res);
  } catch { res.writeHead(404); res.end('Not found'); }
});
// HTTP serves only the public CA, never the app, private keys, or API.
const download = http.createServer((req,res) => {
  if (req.url !== '/happy-aging-dev-ca.crt') { res.writeHead(404); res.end('Use /happy-aging-dev-ca.crt to download the development CA.'); return; }
  res.writeHead(200, {'Content-Type':'application/x-x509-ca-cert','Content-Disposition':'attachment; filename="happy-aging-dev-ca.crt"'});
  res.end(readFileSync(resolve(certificates,'ca.crt')));
});
for (const service of [server,download]) service.on('error', error => { console.error(error.message); process.exit(1); });
server.listen(5443,host,()=>console.log(`Phone app: https://${host}:5443`));
download.listen(5080,host,()=>console.log(`Public CA: http://${host}:5080/happy-aging-dev-ca.crt`));
process.on('SIGINT',()=>{ server.close(); download.close(); });
