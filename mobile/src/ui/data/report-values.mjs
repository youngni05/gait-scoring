export const directionLabel = direction => ({ image_left: "화면 왼쪽으로 걷기", image_right: "화면 오른쪽으로 걷기", unknown: "진행 방향 미확인" })[direction] ?? direction;

const finite = value => typeof value === "number" && Number.isFinite(value);

export function prepareKneeSeries(segment, timeBySegment) {
  const times = timeBySegment?.[segment.segment_id];
  const curves = segment.knee?.views?.supported?.curve_deg;
  const indices = segment.observation?.source_indices;
  const unavailable = reason => ({ available: false, reason, segment_id: segment.segment_id });
  if (!Array.isArray(times)) return unavailable("이 저장 기록에는 각 프레임의 원본 시각이 없습니다. 원본 시각을 포함해 다시 분석한 기록에서 그래프를 볼 수 있습니다.");
  if (times.length < 2 || !curves || ![curves.left, curves.right, indices].every(a => Array.isArray(a) && a.length === times.length)) return unavailable("원본 시각·각도·프레임 배열의 길이가 맞지 않아 그래프를 보류했습니다.");
  if (!times.every((t, i) => finite(t) && (i === 0 || t > times[i - 1])) || !indices.every((n, i) => Number.isInteger(n) && (i === 0 || n > indices[i - 1]))) return unavailable("원본 시각이나 프레임 순서를 확인하지 못해 그래프를 보류했습니다.");
  if (!Object.values(curves).every(a => Array.isArray(a) && a.every(v => v === null || finite(v)))) return unavailable("각도 배열에 확인할 수 없는 값이 있어 그래프를 보류했습니다.");
  const origin = times[0];
  return { available: true, segment_id: segment.segment_id, direction: segment.direction, method_id: segment.knee.method_id,
    source_time_ms: times, source_indices: indices, origin_ms: origin, axis: "elapsed_seconds", unit: "deg",
    x: times.map(t => (t - origin) / 1000), left: curves.left, right: curves.right };
}

export function reportMetric(segment, report, id) {
  return report?.segments?.find(row => row.segment_id === segment.segment_id)?.metrics?.find(metric => metric.id === id);
}

// Summarize one recording. Each value keeps the interval in which that leg was seen.
function sideObservation(segment, side, evidence, localRows, settings) {
  const count = evidence.source_frame_count;
  const seen = evidence.coverage?.[side];
  if (!seen || seen.valid_frames < 2) return null;
  const full = evidence.source_frames_contiguous === true && seen.valid_frames === count;
  const minFrames = settings?.min_frames ?? 7, minSpan = settings?.min_span_ms ?? 500;
  const local = localRows?.find(row => row.side === side && row.segment_id === segment.segment_id &&
    row.direction === segment.direction && row.unit === "deg" && row.status === "limited_local_observation" &&
    Number.isInteger(row.valid_count) && row.valid_count >= minFrames && row.valid_count <= seen.valid_frames &&
    finite(row.value) && finite(row.start_ms) && finite(row.stop_ms) && row.stop_ms - row.start_ms >= minSpan);

  let value = evidence.supported_range_deg?.[side], frames = seen.valid_frames;
  let start = seen.first_source_ms, stop = seen.last_source_ms;
  let isolatedGaps = false;
  if (!full && local) {
    ({value, valid_count:frames, start_ms:start, stop_ms:stop} = local);
  } else if (!full) {
    // Only real angles enter the existing percentile range. Trim missing ends
    // and allow isolated missing samples inside a time-contiguous pass.
    const curve = segment.knee?.views?.supported?.curve_deg?.[side];
    if (!evidence.source_frames_contiguous || frames < minFrames || stop - start < minSpan ||
        !Array.isArray(curve) || curve.length !== count) return null;
    const observed = curve.flatMap((angle, index) => finite(angle) ? [index] : []);
    if (observed.length !== frames || observed.some((index, i) => i > 0 && index - observed[i - 1] > 2)) return null;
    isolatedGaps = observed.at(-1) - observed[0] + 1 > frames;
  }
  if (!finite(value) || !finite(start) || !finite(stop) || stop <= start) return null;
  return { side, value, full, isolatedGaps, frames, start, stop, duration: stop - start,
    segmentId: segment.segment_id, direction: segment.direction };
}

export function visitKneeSummary(visit) {
  const { segments, current_report, segmentation } = visit.analysis;
  const sides = ["left", "right"];
  const candidates = { left: [], right: [] };
  const pairs = [];
  const rank = (a, b) => Number(b.full) - Number(a.full) || Number(a.isolatedGaps) - Number(b.isolatedGaps) ||
    b.duration - a.duration || b.frames - a.frames || a.start - b.start;
  for (const segment of segments) {
    if (!["image_left", "image_right"].includes(segment.direction)) continue;
    const id = segment.segment_id;
    const evidence = reportMetric(segment, current_report, "knee")?.evidence;
    const count = evidence?.source_frame_count;
    if (!Number.isInteger(count) || count < 2) continue;
    const fullPair = [];
    for (const side of sides) {
      const row = sideObservation(segment, side, evidence, visit.local_observations_by_segment?.[id], segmentation?.settings);
      if (!row) continue;
      candidates[side].push(row);
      if (row.full) fullPair.push(row);
    }
    if (fullPair.length === 2) pairs.push(fullPair);
  }
  if (pairs.length) return pairs.sort((a, b) => rank(a[0], b[0]))[0];
  return sides.flatMap(side => candidates[side].sort(rank).slice(0, 1));
}

export function cardMetric(segment, report, kind) {
  const metric = kind === "cadence" ? segment.gait?.metrics?.cadence : segment.gait?.metrics?.left_minus_right_step_time;
  const narrative = reportMetric(segment, report, kind === "cadence" ? "rhythm" : "step_time");
  const unit = kind === "cadence" ? "steps/min" : "s";
  const withheld = reason => ({ value: null, unit, reason, status: "withheld" });
  if (!narrative || narrative.status === "withheld") return withheld(narrative?.sentence ?? "이 기록에는 표시 가능 여부를 확인할 관측 설명이 없습니다.");
  if (segment.direction === "unknown") return withheld("직선 구간의 진행 방향을 확인하지 못했습니다.");
  if (!metric || metric.unit !== unit || !finite(metric.value)) return withheld(narrative.sentence || "이 구간의 측정값을 확인하지 못했습니다.");
  if (kind !== "cadence") {
    const events = segment.gait.events ?? [], sides = segment.gait.metrics.step_time;
    if (!["left", "right"].every(side => events.some(e => e.side === side) && sides?.[side]?.valid_count >= 2 && finite(sides[side].value))) return withheld("같은 직선 구간에서 양발 접촉과 각 발의 유효 간격을 충분히 확인하지 못했습니다.");
  }
  return { value: metric.value, unit, status: narrative.status, sentence: narrative.sentence, valid_count: metric.valid_count };
}
