/** Durable outbox. Commit identical request bytes before the first network send. */
export interface PendingCapture { key: string; id: string; subjectId: string; fileName: string; body: string; delivery?: 'unsent' | 'unconfirmed' | 'rejected'; rejectionStatus?: number }

function database(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('gait-pending-captures', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('pending', { keyPath: 'key' });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(new Error('저장 요청을 기기에 보관하지 못했습니다. 저장 공간과 브라우저 설정을 확인하세요.'));
  });
}

export interface CaptureLease { key: string; owner: string }

async function transaction<T>(action: (table: IDBObjectStore, finish: (value: T) => void) => void, lease?: CaptureLease): Promise<T> {
  const db = await database();
  try {
    return await new Promise<T>((resolve, reject) => {
      const tx = db.transaction('pending', 'readwrite');
      let value: T;
      tx.oncomplete = () => resolve(value);
      tx.onabort = () => reject(tx.error ?? new Error('저장 요청을 기기에 보관하지 못했습니다.'));
      const table = tx.objectStore('pending');
      const run = () => action(table, v => { value = v; });
      if (!lease) run();
      else {
        const q = table.get(lease.key);
        q.onsuccess = () => {
          if (q.result?.owner !== lease.owner || q.result.expires <= Date.now()) { tx.abort(); return; }
          run();
        };
      }
    });
  } finally { db.close(); }
}

export function readPending(key: string): Promise<PendingCapture | null> {
  return transaction((table, finish) => {
    const request = table.get(key);
    request.onsuccess = () => finish(request.result ?? null);
  });
}

/** Competing tabs reuse the first durable request, never overwrite an uncertain send. */
export function keepPending(value: PendingCapture, lease?: CaptureLease): Promise<PendingCapture> {
  return transaction((table, finish) => {
    const request = table.get(value.key);
    request.onsuccess = () => {
      if (request.result) finish(request.result);
      else { table.add(value); finish(value); }
    };
  }, lease);
}

export function clearPending(key: string, id: string, lease?: CaptureLease): Promise<void> {
  return transaction((table, finish) => {
    const request = table.get(key);
    request.onsuccess = () => {
      if (request.result?.id === id) table.delete(key);
      finish(undefined);
    };
  }, lease);
}

/** A durable, fenced lease prevents queued callers from creating a second capture.
 * Expired work is never assumed rejected. Takeover conservatively preserves its body.
 */
const LEASE_MS = 90000;
export async function withPendingLock<T>(key: string, action: (lease: CaptureLease) => Promise<T>): Promise<T> {
  const run = async () => {
    const lease = {key:'!capture-lock:'+key, owner:crypto.randomUUID()};
    const acquired = await transaction<boolean>((table, finish) => {
      const q = table.get(lease.key);
      q.onsuccess = () => {
        if (q.result && q.result.expires > Date.now()) { finish(false); return; }
        if (q.result) {
          const old = table.get(key);
          old.onsuccess = () => { if (old.result) table.put({...old.result, delivery:'unconfirmed', rejectionStatus:undefined}); };
        }
        table.put({...lease, expires:Date.now()+LEASE_MS}); finish(true);
      };
    });
    if (!acquired) throw Error('다른 화면에서 이 대상자의 분석을 처리하고 있습니다. 잠시 후 결과를 확인해 주세요.');
    const heartbeat = window.setInterval(() => {
      void transaction<void>((table, finish) => {
        table.put({...lease, expires:Date.now()+LEASE_MS}); finish(undefined);
      }, lease).catch(() => {}); // All later writes are fenced even if renewal fails.
    }, 20000);
    try { return await action(lease); }
    finally {
      clearInterval(heartbeat);
      await transaction<void>((table, finish) => {
        const q = table.get(lease.key);
        q.onsuccess = () => { if (q.result?.owner === lease.owner) table.delete(lease.key); finish(undefined); };
      });
    }
  };
  if (!navigator.locks) return run();
  return navigator.locks.request('gait-capture:'+key, {ifAvailable:true}, lock => {
    if (!lock) throw Error('다른 화면에서 이 대상자의 분석을 처리하고 있습니다. 잠시 후 결과를 확인해 주세요.');
    return run();
  });
}

export function checkLease(lease: CaptureLease): Promise<void> {
  return transaction((_table, finish) => finish(undefined), lease);
}

export function markDelivery(key: string, id: string, delivery: PendingCapture['delivery'], rejectionStatus?: number, lease?: CaptureLease): Promise<void> {
  return transaction((table, finish) => {
    const request = table.get(key);
    request.onsuccess = () => {
      if (request.result?.id !== id) { table.transaction.abort(); return; }
      table.put({...request.result, delivery, rejectionStatus});
      finish(undefined);
    };
  }, lease);
}
