# Problem: Optimal Label Placement for Line Charts

## Background

When visualizing data with many points, labeling each point is essential for readability. But naive label placement creates visual clutter: labels overlap each other, cover data points, intersect connection lines, or get clipped by canvas edges. This is a well-known problem in information visualization with real research behind it (force-directed layouts, greedy displacement, integer programming approaches).

The challenge: given a set of data points connected by lines and rectangular label boxes, find positions for the labels that minimize all sources of visual clutter simultaneously.

## Objective

Implement `place_labels(canvas_w, canvas_h, points_x, points_y, label_w, label_h)` that returns label center positions optimizing a composite score of:

1. **Line overlap** — labels should not intersect the stepped lines connecting consecutive data points (horizontal-then-vertical, 'before' stepping; heavily penalized)
2. **Point overlap** — labels should not cover their own data point (radius 8px)
3. **Label-label overlap** — labels should not overlap each other
4. **Canvas clipping** — labels should stay within canvas bounds
5. **Proximity** — labels should be close to their assigned data point
6. **Stability** — label positions should remain consistent when canvas size changes (important for responsive design)

## Files You Can Modify

- `experiment/placer.py` — the label placement implementation

You may **NOT** modify:
- `eval.py` — the evaluation harness

## Interface

Your function must have this exact signature:

```python
def place_labels(canvas_w, canvas_h, points_x, points_y, label_w, label_h):
    """Place label boxes near data points to minimize visual clutter.

    Args:
        canvas_w: Canvas width in pixels (int)
        canvas_h: Canvas height in pixels (int)
        points_x: List of x positions of data points in pixels (list of float)
        points_y: List of y positions of data points in pixels (list of float)
        label_w: Width of each label box in pixels (int)
        label_h: Height of each label box in pixels (int)

    Returns:
        List of (x, y) tuples representing label box centers in pixels.
        Must have the same length as points_x/points_y.
    """
```

## Test Configuration

- **5 random configurations**, each with **10 data points**
- **Y values are always monotonically increasing or decreasing** (randomly chosen per config)
- **X positions** are randomly spread with minimum 30px spacing between consecutive points
- **Label boxes**: 100x40 pixels each
- **Baseline canvas**: 800x600 pixels
- **Stability test**: 5 additional canvas sizes (600x450, 1000x750, 400x300, 1200x900, 1400x1000)
- The final score is the **average** across all 5 configurations
- Random seed is fixed at 42 for reproducibility

## Scoring

```
score = 10.0 * line_overlap + 10.0 * point_overlap + 5.0 * label_overlap
        + 20.0 * clipping - 0.5 * proximity - 0.3 * stability
```

- Overlap/clipping terms: 0 = no issue, 1 = fully overlapped/clipped
- Proximity: 1 = on top of point, 0 = very far
- Stability: 1 = identical positions across all canvas sizes, 0 = highly variable
- **Lower is better** (optimization mode: min)

## Constraints

1. **Libraries**: Only Python standard library. No numpy, no matplotlib, no optimization libraries.
2. **Must handle multiple random configurations**: The score is averaged over 5 randomly generated configs (seed=42).
3. **Must handle arbitrary canvas sizes**: The function is tested on 6 different canvas dimensions.
3. **Must return exactly N labels**: One per data point, no more, no fewer.
4. **Must be deterministic**: Same inputs always produce same outputs.

## Hints

### Priority order (by scoring weight)

The score weights tell you exactly where to focus. Eliminating clipping (weight 20) is worth more than anything else. Line and point overlaps (weight 10 each) come next. Label-label overlap (weight 5) matters moderately. Proximity (−0.5) and stability (−0.3) are small bonuses — worth pursuing only after the penalties are near zero.

### Free wins: stability and clipping

**Stability is free.** Divide all point coordinates by `canvas_w`/`canvas_h` at the start, compute label positions in normalized [0,1] space, then multiply back at the end. If your algorithm is purely a function of these ratios, the stability score will be perfect with zero extra work.

**Clipping is free.** After computing all label centers, clamp each one so the full label box fits inside the canvas: `x = clamp(x, label_w/2, canvas_w - label_w/2)` and the same for y. This guarantees zero clipping penalty — the single highest-weighted term in the score.

### Understanding the stepped-line geometry

Connection lines use "before" stepping: from point A to point B, the line goes **horizontally at A's y-level** to B's x, then **vertically at B's x** to B's y. This creates an L-shaped path. The horizontal segment occupies the y-band of the *source* point, and the vertical segment occupies the x-band of the *destination* point.

Concretely, for consecutive points (x₁,y₁) → (x₂,y₂):
- Horizontal segment: y = y₁, spanning x₁ to x₂
- Vertical segment: x = x₂, spanning y₁ to y₂

A label placed *above* point i avoids the horizontal segment leaving point i (which extends rightward at y_i). A label placed *below* would sit right on it.

### Exploiting monotonic y-values

The y-values are always monotonically increasing or decreasing. Detect which direction using `points_y[0] < points_y[-1]`. This determines which side of each point is "safer" for label placement:
- **Y increasing** (points trend downward on screen): the stepped lines extend rightward at each point's y-level and then drop down. Placing labels *above* their points (lower y value, i.e., `y - offset`) keeps them away from the horizontal line segments.
- **Y decreasing** (points trend upward on screen): the opposite — placing labels *below* (higher y value) is generally safer.

### Avoiding point overlap

The point overlap check expands the label rectangle by the point radius (8px) in every direction. So to avoid covering your own data point, the label center must be at least `label_h/2 + 8` pixels away vertically (28px for 40px labels) or `label_w/2 + 8` pixels away horizontally (58px for 100px labels). Vertical offset is cheaper in terms of proximity.

### Suggested approach: candidate positions + greedy assignment

1. **Generate candidates**: For each point, compute 4–8 candidate label positions (above, below, above-left, above-right, etc.) at a fixed offset distance.
2. **Score each candidate** against already-placed labels and all line segments. The eval code checks every label against every line segment — not just the adjacent ones — so your scoring should too.
3. **Place greedily** (e.g., left to right): for each point, pick the candidate with the lowest cost considering already-placed labels.
4. **Refine**: After initial placement, do a few passes of local optimization — try moving each label to its other candidates and keep the move if total score improves.
5. **Clamp** all positions to canvas bounds at the very end.

### Common pitfalls

- Labels are 100×40px — they're wide. Two labels that are adjacent in x with only 30–40px spacing will overlap horizontally even if they're at different y offsets. You may need to stagger them vertically.
- The line overlap check tests *all* labels against *all* line segments, not just label i against segment i. A label placed to avoid its own segment might land on a distant segment.
- Proximity uses `tanh(avg_normalized_dist / 0.05)` — the bonus saturates quickly. Labels more than ~50px away from their point get diminishing returns from moving closer, so don't sacrifice overlap avoidance for proximity.
