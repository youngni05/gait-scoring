const CACHE = 'team-gait-shell-v10';
const SHELL = ['/', '/manifest.webmanifest', '/icon-192.png', '/icon-512.png', '/icon-maskable-512.png', '/apple-touch-icon.png', '/favicon.svg', '/fonts/wanted-sans/WantedSansVariable.woff2'];
self.addEventListener('install', event => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    await cache.addAll(SHELL);
    const html = await (await cache.match('/')).text();
    const bundles = [...html.matchAll(/(?:src|href)="([^"]*assets\/[^"]+)"/g)].map(m => new URL(m[1], self.location.origin).href);
    await cache.addAll(bundles);
    await self.skipWaiting();
  })());
});
self.addEventListener('activate', event => event.waitUntil((async () => {
  for (const name of await caches.keys()) if (name.startsWith('team-gait-shell-') && name !== CACHE) await caches.delete(name);
  await self.clients.claim();
})()));
self.addEventListener('fetch', event => {
  const u = new URL(event.request.url);
  if (event.request.method !== 'GET' || u.origin !== self.location.origin) return;
  if (event.request.mode === 'navigate' && u.pathname === '/') {
    event.respondWith(fetch(event.request).catch(() => caches.match('/')));
  } else if (u.pathname.startsWith('/assets/') || SHELL.includes(u.pathname)) {
    event.respondWith((async () => {
      const cache = await caches.open(CACHE);
      const hit = await cache.match(event.request);
      if (hit) return hit;
      const res = await fetch(event.request);
      if (res.ok) await cache.put(event.request, res.clone());
      return res;
    })());
  }
});
