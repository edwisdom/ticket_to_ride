<!--
  The board's static layer: frame and score track, sea, land, water, relief and lettering.
  It never changes during a game, so it is its own SVG and is never repainted by a hover.
-->
<script lang="ts">
  import type { Layout } from './layout';
  import { SWATCH } from './palette';

  let { layout, routePoints }: { layout: Layout; routePoints: number[] } = $props();

  const [W, H] = $derived(layout.size);
  const s = $derived(layout.square);
  const id = `bg-${Math.random().toString(36).slice(2, 8)}`;
  const base = $derived(layout.basemap);
  const compass = $derived(layout.decor.compass);
  const legend = $derived(layout.decor.legend);
</script>

<svg class="layer" viewBox="0 0 {W} {H}" aria-hidden="true">
  <defs>
    <radialGradient id="{id}-sea" gradientUnits="userSpaceOnUse" cx={W * 0.45} cy={H * 0.45} r={W * 0.62}>
      <stop offset="0" stop-color="#a9c6d3" />
      <stop offset="0.7" stop-color="#8fb2c4" />
      <stop offset="1" stop-color="#6f94aa" />
    </radialGradient>
    <radialGradient id="{id}-vignette" gradientUnits="userSpaceOnUse" cx={W / 2} cy={H / 2} r={W * 0.58}>
      <stop offset="0.55" stop-color="#5a3a17" stop-opacity="0" />
      <stop offset="1" stop-color="#5a3a17" stop-opacity="0.32" />
    </radialGradient>
    <!-- Parchment: broad mottling and a fine grain, both from noise. -->
    <filter id="{id}-paper" x="0" y="0" width="100%" height="100%">
      <feTurbulence type="fractalNoise" baseFrequency="0.006" numOctaves="4" seed="11" result="blotch" />
      <feColorMatrix
        in="blotch"
        type="matrix"
        values="0 0 0 0 0.55  0 0 0 0 0.42  0 0 0 0 0.22  0 0 0 -1.3 0.62"
        result="blotchTint"
      />
      <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="3" result="grain" />
      <feColorMatrix
        in="grain"
        type="matrix"
        values="0 0 0 0 0.35  0 0 0 0 0.25  0 0 0 0 0.12  0 0 0 -0.9 0.5"
        result="grainTint"
      />
      <feMerge>
        <feMergeNode in="SourceGraphic" />
        <feMergeNode in="blotchTint" />
        <feMergeNode in="grainTint" />
      </feMerge>
      <feComposite in2="SourceGraphic" operator="in" />
    </filter>
    <filter id="{id}-soft" x="-5%" y="-5%" width="110%" height="110%">
      <feGaussianBlur stdDeviation="1.2" />
    </filter>
    <clipPath id="{id}-map">
      <rect x={s} y={s} width={W - 2 * s} height={H - 2 * s} />
    </clipPath>
    <symbol id="{id}-peak" viewBox="-8 -6 16 10" width="16" height="10" overflow="visible">
      <path d="M-7 3.6L-2.2-3.4L0.4 0.2L2.4-2.2L7 3.6Z" fill="#cdb387" />
      <path d="M-2.2-3.4L-0.6 3.6H-7ZM2.4-2.2L3.4 3.6H0.9Z" fill="#a48455" />
      <path d="M-7 3.6L-2.2-3.4L0.4 0.2M1 -0.6L2.4-2.2L7 3.6" fill="none" stroke="#6e5232" stroke-width="0.9" stroke-linejoin="round" />
    </symbol>
  </defs>

  <!-- The frame: oxblood with gilt rules, the score track's squares on top. -->
  <rect width={W} height={H} rx="10" fill="#5c1d16" />
  <rect x="4" y="4" width={W - 8} height={H - 8} rx="8" fill="none" stroke="#c9a35a" stroke-width="2" />
  <rect x={s - 3} y={s - 3} width={W - 2 * s + 6} height={H - 2 * s + 6} fill="#2c120c" />

  <g clip-path="url(#{id}-map)">
    <rect x={s} y={s} width={W - 2 * s} height={H - 2 * s} fill="url(#{id}-sea)" />
    <!-- Water lining along the coasts, the old engravers' way: bands under the land. -->
    <g fill="none" stroke-linejoin="round">
      <path d={base.land} stroke="#cfe0e6" stroke-opacity="0.45" stroke-width="22" />
      <path d={base.land} stroke="url(#{id}-sea)" stroke-width="15" />
      <path d={base.land} stroke="#d6e5ea" stroke-opacity="0.55" stroke-width="10" />
      <path d={base.land} stroke="url(#{id}-sea)" stroke-width="6" />
      <path d={base.land} stroke="#e1ecef" stroke-opacity="0.7" stroke-width="3.5" />
    </g>
    <g filter="url(#{id}-paper)">
      <path d={base.land} fill="#ecdcb2" fill-rule="evenodd" />
    </g>
    <path d={base.states} fill="none" stroke="#9c825c" stroke-width="0.8" stroke-dasharray="1 2.6" stroke-linecap="round" opacity="0.55" />
    <path d={base.borders} fill="none" stroke="#8b4b33" stroke-width="1.6" stroke-dasharray="7 3 1.5 3" opacity="0.55" />
    <path d={base.rivers} fill="none" stroke="#6b93b3" stroke-width="1.1" stroke-linejoin="round" stroke-linecap="round" opacity="0.75" />
    <g fill="none" stroke-linejoin="round">
      <path d={base.lakes} stroke="#d6e5ea" stroke-opacity="0.6" stroke-width="5" />
    </g>
    <path d={base.lakes} fill="#9dbfcf" fill-rule="evenodd" stroke="#55788f" stroke-width="0.8" />
    <path d={base.land} fill="none" stroke="#5f6f74" stroke-width="1.1" stroke-linejoin="round" />
    <g opacity="0.75">
      {#each base.peaks as [x, y, k] (`${x},${y}`)}
        <use href="#{id}-peak" x={x - 8 * k} y={y - 5 * k} width={16 * k} height={10 * k} />
      {/each}
    </g>
    {#each layout.decor.water ?? [] as [text, x, y, size, deg] (text)}
      <text class="water" x={x} y={y} font-size={size} transform="rotate({deg} {x} {y})">{text}</text>
    {/each}
    <rect x={s} y={s} width={W - 2 * s} height={H - 2 * s} fill="url(#{id}-vignette)" />
  </g>

  {#if compass}
    {@const r = compass[2]}
    <g class="compass" transform="translate({compass[0]} {compass[1]})">
      <circle r={r * 0.62} fill="none" stroke="#6e5232" stroke-width="1.2" />
      <circle r={r * 0.56} fill="none" stroke="#6e5232" stroke-width="0.6" />
      {#each [0, 45, 90, 135, 180, 225, 270, 315] as deg (deg)}
        {@const long = deg % 90 === 0}
        {@const len = long ? r : r * 0.55}
        <g transform="rotate({deg})">
          <path d="M0 {-len}L{len * 0.16} 0H0Z" fill={long ? '#8e2f22' : '#a48455'} />
          <path d="M0 {-len}L{-len * 0.16} 0H0Z" fill={long ? '#f2e6c8' : '#efe2c0'} stroke="#6e5232" stroke-width="0.6" />
        </g>
      {/each}
      <circle r={r * 0.08} fill="#f2e6c8" stroke="#6e5232" stroke-width="0.8" />
      <text class="compass-n" y={-r - 6} text-anchor="middle">N</text>
    </g>
  {/if}

  {#if legend}
    {@const rows = routePoints.length}
    <!-- The route-points table, in a cartouche: its top-left corner at the decor position. -->
    <g class="legend" transform="translate({legend[0]} {legend[1]}) scale({legend[2]})">
      <rect width="176" height={rows * 16 + 40} rx="5" fill="#f4e8c8" stroke="#5c1d16" stroke-width="1.6" />
      <rect x="4" y="4" width="168" height={rows * 16 + 32} rx="3" fill="none" stroke="#c9a35a" />
      <text class="legend-title" x="88" y="21" text-anchor="middle">Route points</text>
      <line x1="24" y1="28" x2="152" y2="28" stroke="#c9a35a" />
      {#each routePoints as pts, i (i)}
        {@const y = 40 + i * 16}
        {#each { length: i + 1 } as _, k (k)}
          <rect x={14 + k * 19} y={y - 4} width="17" height="8" rx="1.5" fill={SWATCH.gray.fill} stroke="#3b2a1a" stroke-width="0.7" />
        {/each}
        <text class="legend-pts" x="162" y={y + 4.5} text-anchor="end">{pts}</text>
      {/each}
    </g>
  {/if}

  {#each layout.track as [x, y], i (i)}
    <g class="square" class:five={i % 5 === 0}>
      <rect x={x + 3} y={y + 3} width={s - 6} height={s - 6} rx="3" />
      <text x={x + s / 2} y={y + s / 2 + 5.5} text-anchor="middle">{i}</text>
    </g>
  {/each}
</svg>

<style>
  .layer {
    display: block;
    width: 100%;
    height: auto;
  }
  .water {
    font-family: 'Alegreya', serif;
    font-style: italic;
    letter-spacing: 0.32em;
    fill: #3f6580;
    opacity: 0.7;
    text-anchor: middle;
  }
  .compass-n {
    font-family: 'Alegreya SC', serif;
    font-weight: 700;
    font-size: 16px;
    fill: #5c1d16;
  }
  .legend-title {
    font-family: 'Alegreya SC', serif;
    font-weight: 700;
    font-size: 14px;
    fill: #3b2a1a;
  }
  .legend-pts {
    font-family: 'Alegreya Sans', sans-serif;
    font-weight: 700;
    font-size: 13px;
    fill: #3b2a1a;
  }
  .square rect {
    fill: #f1e2bd;
    stroke: #2c120c;
    stroke-width: 1;
  }
  .square.five rect {
    fill: #dcc28a;
  }
  .square text {
    font-family: 'Alegreya Sans', sans-serif;
    font-weight: 700;
    font-size: 15px;
    fill: #4a2414;
  }
</style>
