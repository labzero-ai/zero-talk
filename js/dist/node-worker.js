// The worker thread of Node.js. It gives worker.js the same message port a browser worker has (self.postMessage and self.onmessage), so the
// engine runs the same code in both places.
import { parentPort } from 'node:worker_threads';

globalThis.self = globalThis;
const early = [];
self.postMessage = (message, transfer) => parentPort.postMessage(message, transfer);
parentPort.on('message', (data) => (self.onmessage ? self.onmessage({ data }) : early.push(data)));
await import('./worker.js');
for (const data of early.splice(0)) self.onmessage({ data });
