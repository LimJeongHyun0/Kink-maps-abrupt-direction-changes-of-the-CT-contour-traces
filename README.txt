Kink maps: abrupt direction changes of the CT contour traces (all 10 sets)
==========================================================================
Definition
  * trace segments chained into polylines (shared end points, tol 0.02 R)
  * turning angle at every joint between consecutive straight elements (0 = straight, 90 = right angle)
  * along the line: Gaussian spread of each joint's turning angle (sigma 0.06 R)  -> local turning intensity
  * colour: 0 deg -> blue (HSB hue 225), >= 60 deg -> red (hue 0); S 0.8, B 1.0
  * surface: adaptive-kernel local mean of the line values (hot spots = cusps)

lines/         kink_lines_<set>.png (annotated) / kink_lines_plain_<set>.png / kink_lines_marked_<set>.png (rings at cusps >= 60)
filled/        kink_fill_<set>.png                fully filled disk (main)
filled_faded/  kink_fill_faded_<set>.png          faded to white where no trace lies nearby (colours there are extrapolated)
qa/            qa_kink_fill_<set>.png             surface with the polylines overlaid
kink_metrics.csv    per set: joints, cusps (>=45 / >=30), trace length, turning per R, cusps per R, length fraction with turning > 30
composite_kink_filled_faded.png                   composite of the faded variant
scripts/       parse_traces.py -> circle_map.py -> kink_maps.py -> kink_composites.py

Notes: joints closer than 0.01 R are merged; values follow the granularity of the manual tracing.
