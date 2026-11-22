"""Read projections for the UI; no inference or model parameter changes."""
import math


def legacy_curves(result):
    curves, missing = {}, {}
    for joint in ("hip", "knee", "ankle"):
        graph = getattr(result, joint + "_graph_json", None) if result else None
        if (isinstance(graph, dict) and graph.get("schema_version") == "gait-cycle-degrees-1"
                and all(isinstance(graph.get(side), list) and len(graph[side]) == 51
                        and all(type(v) in (int, float) and math.isfinite(v) for v in graph[side])
                        for side in ("left", "right"))):
            curves[joint] = {side: graph[side] for side in ("left", "right")}
        else:
            missing[joint] = "저장된 그래프가 없습니다." if graph is None else "그래프 축·단위가 확인되지 않아 표시하지 않습니다."
    return curves, missing


# Project only the fields used by the existing visitGdi/walkingEstimate functions.
# Curve arrays, raw coordinates and model provenance never leave PostgreSQL here.
SUMMARY_SQL = """
SELECT m.id, m.participant_id, m.age, m.analysis_status,
       coalesce(m.analysis_started_at, m.measured_at) AS measured_at,
       CASE WHEN m.analysis_started_at IS NOT NULL THEN 'analysis_start'
            WHEN m.analysis_status IS NULL THEN 'legacy_capture' ELSE 'capture' END AS date_basis,
       m.processing_metadata->'height_cm' AS height_cm,
       m.processing_metadata->>'file_name' AS file_name,
       r.gdi_score, r.cadence_spm,
       CASE WHEN r.report_json IS NULL THEN NULL ELSE jsonb_build_object(
         'local_step_intervals_by_segment', r.report_json->'local_step_intervals_by_segment',
         'analysis', jsonb_build_object(
           'segments', coalesce((SELECT jsonb_agg(jsonb_build_object(
             'segment_id', s->'segment_id', 'direction', s->'direction',
             'start_ms', s->'start_ms', 'stop_ms', s->'stop_ms',
             'gdi', jsonb_build_object('models', jsonb_build_object('normal100', jsonb_build_object(
               'status', s#>'{gdi,models,normal100,status}', 'value', s#>'{gdi,models,normal100,value}'))),
             'gait', jsonb_build_object('steps', s#>'{gait,steps}', 'quality_reasons', s#>'{gait,quality_reasons}')
           ) ORDER BY ordinal) FROM jsonb_array_elements(coalesce(r.report_json#>'{analysis,segments}', '[]'))
             WITH ORDINALITY AS segments(s, ordinal)), '[]'),
           'current_report', jsonb_build_object('segments', coalesce((SELECT jsonb_agg(jsonb_build_object(
             'segment_id', c->'segment_id',
             'metrics', jsonb_path_query_array(c, '$.metrics[*] ? (@.id == "rhythm")')
           ) ORDER BY ordinal) FROM jsonb_array_elements(coalesce(r.report_json#>'{analysis,current_report,segments}', '[]'))
             WITH ORDINALITY AS reports(c, ordinal)), '[]'))
         )
       ) END AS preview
FROM gait_measurements m
LEFT JOIN gait_analysis_results r ON r.measurement_id = m.id
WHERE m.participant_id = :participant_id
ORDER BY coalesce(m.received_at, m.analysis_started_at, m.measured_at) DESC, m.id
LIMIT :limit OFFSET :offset
"""
