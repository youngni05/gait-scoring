// Read-only DB export. Node file descriptors preserve binary bytes on Windows.
import { execFileSync } from 'node:child_process';
import { closeSync, mkdirSync, openSync, renameSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const directory = join(root, 'backups');
const compose = ['compose', '-f', 'compose.yaml'];
execFileSync('docker', [...compose, 'exec', '-T', 'db', 'pg_isready'], { cwd: root, stdio: 'inherit' });
mkdirSync(directory, { recursive: true, mode: 0o700 });
const destination = join(directory, `gait-${new Date().toISOString().replace(/[:.]/g, '-')}.dump`);
const partial = destination + '.partial';
let fd = openSync(partial, 'wx', 0o600);
try {
  execFileSync('docker', [...compose, 'exec', '-T', 'db', 'sh', '-c',
    'exec pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --format=custom --no-owner --no-acl'],
  { cwd: root, stdio: ['ignore', fd, 'inherit'] });
} finally { closeSync(fd); }
fd = openSync(partial, 'r');
try {
  execFileSync('docker', [...compose, 'exec', '-T', 'db', 'pg_restore', '--list'],
    { cwd: root, stdio: [fd, 'ignore', 'inherit'] });
} finally { closeSync(fd); }
renameSync(partial, destination);
console.log(`Archive created and catalog checked: ${destination}`);
console.log('A restore rehearsal is still required. Keep this private; it contains account hashes and participant data.');
