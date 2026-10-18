<!--
  A train card: a colour's railcar on a sunburst in that colour, the colour's symbol in the
  corners, its name below. The locomotive rides on all eight colours.
-->
<script lang="ts">
  import type { CardColor } from '../board/layout';
  import { CARD_COLORS, SWATCH, SYMBOL } from '../board/palette';
  import Railcar from './Railcar.svelte';

  interface Props {
    color: CardColor | 'locomotive';
    /** Shown as a badge when more than one. */
    count?: number;
    width?: number;
    /** Drawn faded, e.g. a card that cannot be taken. */
    muted?: boolean;
  }
  let { color, count = 1, width = 72, muted = false }: Props = $props();

  const uid = `card-${Math.random().toString(36).slice(2, 8)}`;
  const sw = $derived(SWATCH[color]);
  const loco = $derived(color === 'locomotive');
</script>

<svg class="card" class:muted viewBox="0 0 100 140" {width} role="img" aria-label="{sw.label}{count > 1 ? ` ×${count}` : ''}">
  <defs>
    <clipPath id="{uid}-panel"><rect x="7" y="7" width="86" height="86" rx="5" /></clipPath>
    <radialGradient id="{uid}-glow" cx="50" cy="78" r="80" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#fff6dc" stop-opacity="0.85" />
      <stop offset="0.5" stop-color="#fff6dc" stop-opacity="0.1" />
      <stop offset="1" stop-color="#000" stop-opacity="0.18" />
    </radialGradient>
  </defs>
  <rect x="1" y="1" width="98" height="138" rx="8" fill="#f4e8c8" stroke="#3b2a1a" stroke-width="1.4" />
  <g clip-path="url(#{uid}-panel)">
    {#if loco}
      {#each CARD_COLORS as c, i (c)}
        <path d="M{-40 + i * 24} 100L{-10 + i * 24} 0H{14 + i * 24}L{-16 + i * 24} 100Z" fill={SWATCH[c].fill} />
      {/each}
    {:else}
      <rect x="7" y="7" width="86" height="86" fill={sw.fill} />
    {/if}
    {#each { length: 12 } as _, i (i)}
      <path d="M50 80L{50 + 120 * Math.cos((i * Math.PI) / 12 + Math.PI)} {80 + 120 * Math.sin((i * Math.PI) / 12 + Math.PI)}L{50 + 120 * Math.cos(((i + 0.5) * Math.PI) / 12 + Math.PI)} {80 + 120 * Math.sin(((i + 0.5) * Math.PI) / 12 + Math.PI)}Z" fill="#fff" opacity="0.1" />
    {/each}
    <rect x="7" y="7" width="86" height="86" fill="url(#{uid}-glow)" />
    <rect x="7" y="80" width="86" height="13" fill="#6f5a3c" />
    <rect x="7" y="80" width="86" height="1.6" fill="#2a1d12" />
    <g transform="translate(8 44) scale(0.84)">
      <Railcar {color} />
    </g>
  </g>
  <rect x="7" y="7" width="86" height="86" rx="5" fill="none" stroke="#3b2a1a" stroke-width="1.2" />
  <circle cx="17" cy="17" r="8" fill="#f4e8c8" stroke="#3b2a1a" stroke-width="0.8" />
  <path d={SYMBOL[color]} transform="translate(17 17) scale(5)" fill={loco ? '#3a2b18' : sw.fill} stroke={sw.dark} stroke-width="0.12" fill-rule="evenodd" />
  <text x="50" y="121" text-anchor="middle" textLength={loco ? 80 : undefined}>{sw.label}</text>
  {#if count > 1}
    <g transform="translate(88 12)">
      <circle r="11" fill="#2b1a10" stroke="#f4e8c8" stroke-width="1.5" />
      <text class="count" y="4.6" text-anchor="middle">{count}</text>
    </g>
  {/if}
</svg>

<style>
  .card {
    display: block;
    height: auto;
    filter: drop-shadow(0 2px 3px rgb(0 0 0 / 0.35));
    transition:
      transform 120ms,
      opacity 120ms;
  }
  .card.muted {
    opacity: 0.45;
  }
  text {
    font-family: 'Alegreya SC', serif;
    font-weight: 700;
    font-size: 15px;
    fill: #3b2a1a;
  }
  .count {
    font-family: 'Alegreya Sans', sans-serif;
    font-size: 14px;
    fill: #f4e8c8;
  }
</style>
