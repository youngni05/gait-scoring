// Run on the destination server. Never overwrite existing configuration.
import { randomBytes } from 'node:crypto';
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const domain = process.argv[2];
if (!domain || !/^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}$/i.test(domain)) {
  throw new Error('Usage: node scripts/init-server-env.mjs api.your-domain.com');
}
const password = randomBytes(32).toString('hex');
const config = [
  `API_DOMAIN=${domain}`,
  'POSTGRES_USER=gaituser',
  'POSTGRES_DB=gaitdb',
  `POSTGRES_PASSWORD=${password}`,
  `DATABASE_URL=postgresql+psycopg://gaituser:${password}@db:5432/gaitdb`,
  `JWT_SECRET_KEY=${randomBytes(48).toString('hex')}`,
  'CORS_ALLOW_ORIGINS=http://127.0.0.1:5173,http://localhost:5173,http://localhost,https://localhost,capacitor://localhost',
  '',
].join('\n');
writeFileSync(fileURLToPath(new URL('../.env.server', import.meta.url)), config, { flag: 'wx', mode: 0o600 });
console.log('Created .env.server. Secrets were not printed. Existing app account passwords are preserved by database restore.');
