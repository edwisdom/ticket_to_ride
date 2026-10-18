<!--
  The board: the static background, then routes, trains, cities and score markers on top.

  Everything game-specific comes in as props, indexed by the engine's segment and seat
  numbers; the board knows no rules. A segment is drawn as its car spaces in the card colour,
  dimmed when closed (the other half of a double, in a 2-3 player game), and covered by trains
  in the owner's colour when claimed: raised, shadowed and coupled, so a claimed route never
  reads as unclaimed spaces of the same colour.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import BoardBackground from './BoardBackground.svelte';
  import type { BoardModel, Segment } from './layout';
  import { couplings, plaqueRect, scoreMarkers } from './geometry';
  import { SEAT_COLORS, SWATCH, SYMBOL, seatColor } from './palette';
  import { textWidth } from './measure';

  interface Props {
    model: BoardModel;
    /** Per segment, the seat that claimed it. */
    owner?: (number | null)[];
    /** Per segment, whether it is a closed half of a double. */
    closed?: boolean[];
    /** Per seat, the score on the track. */
    scores?: number[];
    /** Cities to point out, e.g. a hovered ticket's. */
    highlight?: string[];
    /** Segments that can be clicked; none when absent. */
    selectable?: Set<number> | null;
    onselect?: (segment: number) => void;
    onhover?: (segment: number | null) => void;
  }

  let {
    model,
    owner = [],
    closed = [],
    scores = [],
    highlight = [],
    selectable = null,
    onselect,
    onhover,
  }: Props = $props();

  const layout = $derived(model.layout);
  const [W, H] = $derived(layout.size);
  const CW = $derived(layout.car_width);
  // Ids inside an inline SVG are document-wide, so each board gets its own.
  const id = `fg-${Math.random().toString(36).slice(2, 8)}`;
  const font = $derived(`500 ${layout.label_size}px "Alegreya SC"`);

  let fontsLoaded = $state(false);
  onMount(() => {
    document.fonts.load(font).then(() => (fontsLoaded = true));
  });
  const plaques = $derived.by(() => {
    void fontsLoaded;
    return new Map(
      layout.cities.map((c) => [
        c.name,
        plaqueRect(
          c.label,
          textWidth(c.name, font) + c.name.length * layout.label_tracking * layout.label_size,
          layout.label_size,
          layout.label_pad,
        ),
      ]),
    );
  });

  let hovered = $state<number | null>(null);
  const hot = $derived(new Set(highlight));
  const hoveredSeg = $derived(hovered === null ? null : model.segments[hovered]);

  function hover(seg: number | null) {
    hovered = seg;
    onhover?.(seg);
  }

  const claimed = $derived(model.segments.filter((s) => owner[s.index] != null));
  const markers = $derived(scoreMarkers(scores, layout.track, layout.square));

  function tip(seg: Segment) {
    const mid = seg.cars[Math.floor(seg.cars.length / 2)]!;
    const points = model.routePoints[seg.length - 1] ?? 0;
    const text = `${seg.a} – ${seg.b}`;
    const sub = `${seg.length} ${SWATCH[seg.color].label.toLowerCase()} · ${points} point${points === 1 ? '' : 's'}`;
    const w = Math.max(textWidth(text, '600 14px "Alegreya Sans"'), textWidth(sub, '400 13px "Alegreya Sans"')) + 20;
    const x = Math.min(Math.max(mid[0] - w / 2, layout.square + 4), W - layout.square - w - 4);
    const y = mid[1] > H / 2 ? mid[1] - 64 : mid[1] + 18;
    return { x, y, w, text, sub };
  }
</script>

<div class="board" style="aspect-ratio: {W} / {H}">
  <BoardBackground {layout} routePoints={model.routePoints} />
  <svg class="layer" viewBox="0 0 {W} {H}" role="group" aria-label="{layout.map} board">
    <defs>
      {#each SEAT_COLORS as c, seat (seat)}
        <linearGradient id="{id}-train-{seat}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color={c.light} />
          <stop offset="0.45" stop-color={c.fill} />
          <stop offset="1" stop-color={c.dark} />
        </linearGradient>
      {/each}
      <filter id="{id}-shadow" x="-30%" y="-60%" width="160%" height="220%">
        <feGaussianBlur stdDeviation="1.6" />
      </filter>
      <filter id="{id}-glow" x="-100%" y="-100%" width="300%" height="300%">
        <feGaussianBlur stdDeviation="4" />
      </filter>
    </defs>

    <!-- Unclaimed car spaces. -->
    {#each model.segments as seg (seg.index)}
      {@const sw = SWATCH[seg.color]}
      {@const isClosed = closed[seg.index] ?? false}
      {@const canPick = selectable?.has(seg.index) ?? false}
      <g
        class="segment"
        class:closed={isClosed}
        class:pickable={canPick}
        class:dimmed={selectable !== null && !canPick && owner[seg.index] == null}
        class:hovered={hovered === seg.index}
        role="button"
        tabindex={canPick ? 0 : -1}
        aria-disabled={!canPick}
        aria-label="{seg.a} to {seg.b}, {seg.length} {sw.label}"
        onmouseenter={() => hover(seg.index)}
        onmouseleave={() => hover(null)}
        onclick={() => canPick && onselect?.(seg.index)}
        onkeydown={(e) => canPick && (e.key === 'Enter' || e.key === ' ') && onselect?.(seg.index)}
      >
        {#each seg.cars as [x, y, deg], k (k)}
          {@const L = seg.carLength}
          <g transform="translate({x} {y}) rotate({deg})">
            <rect class="space" x={-L / 2} y={-CW / 2} width={L} height={CW} rx="2.2" fill={sw.fill} stroke={sw.dark} />
            <rect class="inset" x={-L / 2 + 1.6} y={-CW / 2 + 1.6} width={L - 3.2} height={CW - 3.2} rx="1.2" />
            {#if seg.color !== 'gray'}
              <path d={SYMBOL[seg.color]} transform="rotate({-deg}) scale({CW * 0.27})" fill={sw.ink} fill-rule="evenodd" />
            {/if}
          </g>
        {/each}
      </g>
    {/each}

    <!-- Trains on claimed routes. -->
    {#each claimed as seg (seg.index)}
      {@const seat = owner[seg.index] ?? 0}
      {@const c = seatColor(seat)}
      {@const L = seg.carLength}
      <g class="train" pointer-events="none">
        {#each seg.cars as [x, y, deg], k (k)}
          <rect
            transform="translate({x + 1.8} {y + 2.6}) rotate({deg})"
            x={-L / 2 - 1}
            y={-CW / 2 - 1.5}
            width={L + 2}
            height={CW + 3}
            rx="3"
            fill="#1d1208"
            opacity="0.55"
            filter="url(#{id}-shadow)"
          />
        {/each}
        {#each couplings(seg.cars, L + 2) as [x1, y1, x2, y2], k (k)}
          <line {x1} {y1} {x2} {y2} stroke={c.dark} stroke-width="3" stroke-linecap="round" />
        {/each}
        {#each seg.cars as [x, y, deg], k (k)}
          <g transform="translate({x} {y}) rotate({deg})">
            <rect x={-L / 2 - 1} y={-CW / 2 - 1.5} width={L + 2} height={CW + 3} rx="3" fill="url(#{id}-train-{seat % SEAT_COLORS.length})" stroke={c.dark} stroke-width="1.3" />
            <rect x={-L / 2 + 2.5} y={-CW / 2 + 0.3} width={L - 5} height="2.2" rx="1.1" fill="#fff" opacity="0.38" />
            {#each [-0.25, 0, 0.25] as f (f)}
              <line x1={f * L} y1={-CW / 2 + 0.5} x2={f * L} y2={CW / 2 - 0.5} stroke={c.dark} stroke-width="0.7" opacity="0.55" />
            {/each}
          </g>
        {/each}
      </g>
    {/each}

    <!-- Cities, then their plaques, so a plaque is never under a dot. -->
    {#each layout.cities as c (c.name)}
      <g class="city" class:hot={hot.has(c.name)}>
        {#if hot.has(c.name)}
          <circle class="halo" cx={c.x} cy={c.y} r={layout.city_radius + 9} filter="url(#{id}-glow)" />
        {/if}
        <circle cx={c.x} cy={c.y} r={layout.city_radius} fill="#3a2214" />
        <circle cx={c.x} cy={c.y} r={layout.city_radius - 1.8} fill="#f2e3bf" />
        <circle cx={c.x} cy={c.y} r={layout.city_radius - 3.4} fill="#b8432c" />
      </g>
    {/each}
    {#each layout.cities as c (c.name)}
      {@const p = plaques.get(c.name)!}
      <g class="plaque" class:hot={hot.has(c.name)}>
        <rect x={p.x} y={p.y} width={p.w} height={p.h} rx="2" />
        <text x={p.x + layout.label_pad[0]} y={p.y + layout.label_pad[1] + layout.label_size * 0.8} font-size={layout.label_size}>{c.name}</text>
      </g>
    {/each}

    {#each markers as m (m.seat)}
      {@const c = seatColor(m.seat)}
      <g class="marker" transform="translate({m.x} {m.y})">
        <ellipse cx="1.5" cy="3" rx="13" ry="11" fill="#000" opacity="0.35" filter="url(#{id}-shadow)" />
        <circle r="12" fill="url(#{id}-train-{m.seat % SEAT_COLORS.length})" stroke={c.dark} stroke-width="1.5" />
        <circle r="5" fill="#fff" opacity="0.25" cx="-3" cy="-4" />
      </g>
    {/each}

    {#if hoveredSeg}
      {@const t = tip(hoveredSeg)}
      <g class="tip" pointer-events="none" transform="translate({t.x} {t.y})">
        <rect width={t.w} height="46" rx="5" />
        <text class="tip-title" x="10" y="19">{t.text}</text>
        <text class="tip-sub" x="10" y="37">{t.sub}</text>
      </g>
    {/if}
  </svg>
</div>

<style>
  .board {
    position: relative;
    display: grid;
    width: 100%;
  }
  .board > :global(*) {
    grid-area: 1 / 1;
  }
  .layer {
    display: block;
    width: 100%;
    height: auto;
  }
  .space {
    stroke-width: 1.2;
  }
  .inset {
    fill: none;
    stroke: #fff8e6;
    stroke-width: 0.8;
    opacity: 0.55;
  }
  .segment {
    transition: opacity 120ms;
  }
  .segment.closed {
    opacity: 0.28;
    filter: saturate(0.3);
  }
  .segment.dimmed {
    opacity: 0.45;
  }
  .segment.pickable {
    cursor: pointer;
  }
  .segment.hovered .space {
    stroke: #fff4cf;
    stroke-width: 2.4;
  }
  .segment:focus-visible {
    outline: none;
  }
  .segment:focus-visible .space {
    stroke: #fff4cf;
    stroke-width: 2.4;
  }
  .halo {
    fill: #ffd76a;
    opacity: 0.9;
    animation: pulse 1.4s ease-in-out infinite;
  }
  @keyframes pulse {
    50% {
      opacity: 0.35;
    }
  }
  .plaque rect {
    fill: #f6ecd0;
    stroke: #4a3220;
    stroke-width: 0.9;
  }
  .plaque text {
    font-family: 'Alegreya SC', serif;
    font-weight: 500;
    letter-spacing: 0.03em;
    fill: #2e1d10;
  }
  .plaque.hot rect {
    fill: #ffe08a;
    stroke: #8e2f22;
    stroke-width: 1.6;
  }
  .tip rect {
    fill: #2b1a10;
    opacity: 0.92;
  }
  .tip-title {
    font-family: 'Alegreya Sans', sans-serif;
    font-weight: 600;
    font-size: 14px;
    fill: #fbf1d8;
  }
  .tip-sub {
    font-family: 'Alegreya Sans', sans-serif;
    font-size: 13px;
    fill: #e2cfa6;
  }
</style>
