import { useEffect, useRef, useState } from 'react';
import { dataSource, type Subject, type ResultRecord } from '../data';
import { teamApi } from '../data/teamApi';

type ReportData = [Subject | null, ResultRecord | null];

/** One in-flight operation per mounted report; polling preserves the last response. */
export function useReport(subjectId: string, recordId: string) {
  const [data, setData] = useState<ReportData>();
  const [refreshing, setRefreshing] = useState(true);
  const [retrying, setRetrying] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [retryFailed, setRetryFailed] = useState(false);
  const actions = useRef({ reload: () => {}, retry: () => {} });
  useEffect(() => {
    const abort = new AbortController();
    let timer: number | undefined;
    let busy = false;
    let latest: ReportData | undefined;
    const run = async (retry = false) => {
      if (busy || abort.signal.aborted) return;
      busy = true;
      clearTimeout(timer);
      setRefreshing(true);
      setRetrying(retry);
      setError(null);
      setRetryFailed(false);
      let postFinished = !retry;
      let failed = false;
      try {
        if (retry) {
          await teamApi.shared('/recordings/'+encodeURIComponent(recordId)+'/retry', {method:'POST'}, abort.signal);
          postFinished = true;
          if (abort.signal.aborted) return;
          // Receipt confirms requeue; a subsequent GET failure must not restore a stale failed state.
          if (latest?.[1]) {
            latest = [latest[0], {...latest[1], serverState:'pending'}];
            setData(latest);
          }
        }
        const next: ReportData = latest
          ? [latest[0], await dataSource.getResult(subjectId, recordId, abort.signal)]
          : await Promise.all([dataSource.getSubject(subjectId, abort.signal), dataSource.getResult(subjectId, recordId, abort.signal)]);
        if (!abort.signal.aborted) { latest = next; setData(next); }
      } catch (e) {
        failed = true;
        if (!abort.signal.aborted) {
          if (!postFinished) setRetryFailed(true);
          else setError(e);
        }
      } finally {
        busy = false;
        if (!abort.signal.aborted) {
          setRefreshing(false);
          setRetrying(false);
          if (latest?.[1]?.serverState === 'pending')
            timer = window.setTimeout(() => void run(), failed ? 10000 : 3000);
        }
      }
    };
    actions.current = {reload: () => void run(), retry: () => void run(true)};
    void run();
    return () => { abort.abort(); clearTimeout(timer); };
  }, [subjectId, recordId]);
  return {data, refreshing, retrying, error, retryFailed,
    reload: () => actions.current.reload(), retry: () => actions.current.retry()};
}
