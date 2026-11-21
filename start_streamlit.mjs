import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const rootDir = process.cwd();
const venvStreamlit = path.join(rootDir, '.venv', 'bin', 'streamlit');
const command = fs.existsSync(venvStreamlit) ? venvStreamlit : 'streamlit';

const args = [
  'run',
  'app.py',
  '--server.port=3000',
  '--server.address=0.0.0.0',
  '--server.headless=true',
  '--server.enableCORS=false',
  '--server.enableXsrfProtection=false',
];

const child = spawn(command, args, {
  cwd: rootDir,
  env: process.env,
  stdio: 'inherit',
});

const forwardSignal = (signal) => {
  if (!child.killed) {
    child.kill(signal);
  }
};

process.on('SIGINT', () => forwardSignal('SIGINT'));
process.on('SIGTERM', () => forwardSignal('SIGTERM'));

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});
