/** Durable outbox. Commit identical request bytes before the first network send. */
export interface PendingCapture { key: string; id: string; subjectId: string; fileName: string; body: string }

function database(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('gait-pending-captures', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('pending', { keyPath: 'key' });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(new Error('저장 요청을 기기에 보관하지 못했습니다. 저장 공간과 브라우저 설정을 확인하세요.'));
  });
}

async function transaction<T>(action: (table: IDBObjectStore, finish: (value: T) => void) => void): Promise<T> {
  const db = await database();
  try {
    return await new Promise<T>((resolve, reject) => {
      const tx = db.transaction('pending', 'readwrite');
      let value: T;
      tx.oncomplete = () => resolve(value);
      tx.onabort = () => reject(tx.error ?? new Error('저장 요청을 기기에 보관하지 못했습니다.'));
      action(tx.objectStore('pending'), v => { value = v; });
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
export function keepPending(value: PendingCapture): Promise<PendingCapture> {
  return transaction((table, finish) => {
    const request = table.get(value.key);
    request.onsuccess = () => {
      if (request.result) finish(request.result);
      else { table.add(value); finish(value); }
    };
  });
}

export function clearPending(key: string, id: string): Promise<void> {
  return transaction((table, finish) => {
    const request = table.get(key);
    request.onsuccess = () => {
      if (request.result?.id === id) table.delete(key);
      finish(undefined);
    };
  });
}
