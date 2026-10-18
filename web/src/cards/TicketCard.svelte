<!--
  A destination ticket: a cream stub with a small map of the board, the two cities and the
  line between them, and the points. Hovering it asks the board to point out its cities.
-->
<script lang="ts">
  import type { Layout } from '../board/layout';

  interface Props {
    layout: Layout;
    a: string;
    b: string;
    points: number;
    /** done: connected; failed: the game ended without it. */
    status?: 'open' | 'done' | 'failed';
    /** Trains still needed to connect it, when known. */
    remaining?: number | null;
    width?: number;
    onhover?: (cities: [string, string] | null) => void;
  }
  let { layout, a, b, points, status = 'open', remaining = null, width = 180, onhover }: Props = $props();

  const uid = `ticket-${Math.random().toString(36).slice(2, 8)}`;
  const [W, H] = $derived(layout.size);
  const s = $derived(layout.square);
  // The small map shows the board's map area, fitted into the stub's window.
  const win = { x: 8, y: 8, w: 104, h: 70 };
  const k = $derived(Math.min(win.w / (W - 2 * s), win.h / (H - 2 * s)));
  const ox = $derived(win.x + (win.w - (W - 2 * s) * k) / 2 - s * k);
  const oy = $derived(win.y + (win.h - (H - 2 * s) * k) / 2 - s * k);
  const city = (name: string) => {
    const c = layout.cities.find((c) => c.name === name);
    return c ? { x: ox + c.x * k, y: oy + c.y * k } : { x: 0, y: 0 };
  };
  const pa = $derived(city(a));
  const pb = $derived(city(b));
  // A gentle arc between the two, as a hand-drawn line would be.
  const bow = $derived.by(() => {
    const mx = (pa.x + pb.x) / 2;
    const my = (pa.y + pb.y) / 2;
    const dx = pb.x - pa.x;
    const dy = pb.y - pa.y;
    return { x: mx - dy * 0.18, y: my + dx * 0.18 };
  });
</script>

<svg
  class="ticket"
  class:done={status === 'done'}
  class:failed={status === 'failed'}
  viewBox="0 0 170 112"
  {width}
  role="img"
  aria-label="{a} to {b}, {points} points{status === 'done' ? ', connected' : ''}"
  onmouseenter={() => onhover?.([a, b])}
  onmouseleave={() => onhover?.(null)}
>
  <defs>
    <clipPath id="{uid}-win"><rect x={win.x} y={win.y} width={win.w} height={win.h} rx="3" /></clipPath>
  </defs>
  <path
    d="M6 1H164A5 5 0 0 1 169 6V44A6 6 0 0 0 169 56V106A5 5 0 0 1 164 111H6A5 5 0 0 1 1 106V56A6 6 0 0 0 1 44V6A5 5 0 0 1 6 1Z"
    fill="#f6ecd0"
    stroke="#4a3220"
    stroke-width="1.2"
  />
  <g clip-path="url(#{uid}-win)">
    <rect x={win.x} y={win.y} width={win.w} height={win.h} fill="#a9c6d3" />
    <path d={layout.basemap.thumb} transform="translate({ox} {oy}) scale({k})" fill="#e7d4a4" stroke="#7c8d93" stroke-width={0.8 / k} />
    <path d="M{pa.x} {pa.y}Q{bow.x} {bow.y} {pb.x} {pb.y}" fill="none" stroke="#8e2f22" stroke-width="1.6" stroke-dasharray="3 2" />
    {#each [pa, pb] as p, i (i)}
      <circle cx={p.x} cy={p.y} r="3.2" fill="#b8432c" stroke="#3a2214" stroke-width="1" />
    {/each}
  </g>
  <rect x={win.x} y={win.y} width={win.w} height={win.h} rx="3" fill="none" stroke="#4a3220" stroke-width="1" />
  <g class="points">
    <circle cx="141" cy="31" r="18" fill="#5c1d16" stroke="#c9a35a" stroke-width="2" />
    <text x="141" y="38" text-anchor="middle">{points}</text>
  </g>
  <text class="kind" x="141" y="63" text-anchor="middle">points</text>
  <text class="cities" x="10" y="95">{a}</text>
  <text class="cities" x="10" y="107">{b}</text>
  {#if status === 'done' || status === 'failed'}
    <g class="stamp" transform="translate(141 93) rotate(-10)">
      <rect x="-24" y="-9" width="48" height="18" rx="3" />
      <text y="4.5" text-anchor="middle">{status === 'done' ? 'Done' : 'Failed'}</text>
    </g>
  {:else if remaining !== null}
    <text class="remaining" x="141" y="88" text-anchor="middle">{remaining}</text>
    <text class="kind" x="141" y="100" text-anchor="middle">trains to go</text>
  {/if}
</svg>

<style>
  .ticket {
    display: block;
    height: auto;
    filter: drop-shadow(0 2px 3px rgb(0 0 0 / 0.3));
  }
  .points text {
    font-family: 'Alegreya Sans', sans-serif;
    font-weight: 700;
    font-size: 21px;
    fill: #f6ecd0;
  }
  .kind {
    font-family: 'Alegreya SC', serif;
    font-size: 10px;
    fill: #6a5039;
  }
  .cities {
    font-family: 'Alegreya SC', serif;
    font-weight: 700;
    font-size: 11.5px;
    fill: #2e1d10;
  }
  .remaining {
    font-family: 'Alegreya Sans', sans-serif;
    font-weight: 700;
    font-size: 15px;
    fill: #3b2a1a;
  }
  .stamp rect {
    fill: none;
    stroke-width: 2;
  }
  .stamp text {
    font-family: 'Alegreya SC', serif;
    font-weight: 700;
    font-size: 12px;
  }
  .done .stamp rect {
    stroke: #2f7a3a;
  }
  .done .stamp text {
    fill: #2f7a3a;
  }
  .failed .stamp rect {
    stroke: #a3271d;
  }
  .failed .stamp text {
    fill: #a3271d;
  }
</style>
