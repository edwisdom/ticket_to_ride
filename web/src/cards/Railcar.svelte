<!--
  A railcar in side view, one kind per card colour, drawn in a 100 x 44 box with the rail at
  y = 40. Original art: simple shapes in the manner of a period railway catalogue.
-->
<script lang="ts">
  import type { CardColor } from '../board/layout';
  import { SWATCH } from '../board/palette';

  let { color }: { color: CardColor | 'locomotive' } = $props();

  const sw = $derived(SWATCH[color]);
  const line = '#1f160e';
</script>

{#snippet truck(x: number)}
  <rect x={x - 11} y="32.5" width="22" height="3" rx="1" fill={line} />
  <circle cx={x - 6} cy="36.5" r="3.6" fill={line} />
  <circle cx={x + 6} cy="36.5" r="3.6" fill={line} />
  <circle cx={x - 6} cy="36.5" r="1.3" fill="#8a7a62" />
  <circle cx={x + 6} cy="36.5" r="1.3" fill="#8a7a62" />
{/snippet}

{#snippet frame()}
  <rect x="8" y="30" width="84" height="3.2" fill={line} />
  <rect x="3" y="30.6" width="5" height="2" fill={line} />
  <rect x="92" y="30.6" width="5" height="2" fill={line} />
  {@render truck(24)}
  {@render truck(76)}
{/snippet}

<g class="railcar" stroke-linejoin="round">
  {#if color === 'blue'}
    <!-- Box car: planked sides and a sliding door. -->
    <rect x="9" y="8" width="82" height="22" fill={sw.fill} stroke={sw.dark} stroke-width="1.2" />
    <rect x="7.5" y="6" width="85" height="3" rx="1" fill={sw.dark} />
    {#each [17, 25, 33, 67, 75, 83] as x (x)}
      <line x1={x} y1="9" x2={x} y2="29.5" stroke={sw.dark} stroke-width="0.6" opacity="0.7" />
    {/each}
    <rect x="40" y="10" width="20" height="19" fill="none" stroke={sw.dark} stroke-width="1.2" />
    <line x1="40" y1="10" x2="60" y2="29" stroke={sw.dark} stroke-width="0.8" />
    <line x1="60" y1="10" x2="40" y2="29" stroke={sw.dark} stroke-width="0.8" />
  {:else if color === 'green'}
    <!-- Tank car: a riveted barrel with a dome and a walkway. -->
    <rect x="10" y="12" width="80" height="18" rx="9" fill={sw.fill} stroke={sw.dark} stroke-width="1.2" />
    <rect x="45" y="6" width="10" height="7" rx="1.5" fill={sw.fill} stroke={sw.dark} stroke-width="1.2" />
    <rect x="43.5" y="5" width="13" height="2.2" rx="1" fill={sw.dark} />
    <line x1="14" y1="16" x2="86" y2="16" stroke="#fff" stroke-width="1.6" opacity="0.35" />
    {#each [26, 50, 74] as x (x)}
      <line x1={x} y1="12.5" x2={x} y2="29.5" stroke={sw.dark} stroke-width="0.7" />
    {/each}
  {:else if color === 'black'}
    <!-- Hopper: slanted bins heaped with coal. -->
    <path d="M10 13L13 9H87L90 13Z" fill="#1a1715" />
    <path d="M12 9Q20 3 28 8Q36 2 45 8Q54 3 62 8Q71 2 79 8Q85 5 88 9Z" fill="#141210" />
    <path d="M10 12H90V22L80 30H20L10 22Z" fill={sw.fill} stroke="#5d5853" stroke-width="1.2" />
    {#each [28, 50, 72] as x (x)}
      <line x1={x} y1="12.5" x2={x} y2="29.5" stroke="#5d5853" stroke-width="0.9" />
    {/each}
    <line x1="11" y1="15" x2="89" y2="15" stroke="#8a847c" stroke-width="0.8" />
  {:else if color === 'red'}
    <!-- Caboose: a cupola, windows and railed platforms. -->
    <rect x="16" y="11" width="68" height="19" fill={sw.fill} stroke={sw.dark} stroke-width="1.2" />
    <rect x="14" y="9" width="72" height="3" rx="1" fill={sw.dark} />
    <rect x="39" y="2" width="22" height="8" fill={sw.fill} stroke={sw.dark} stroke-width="1.2" />
    <rect x="37.5" y="0.5" width="25" height="2.5" rx="1" fill={sw.dark} />
    <rect x="42" y="4" width="6" height="4" fill="#f6e3b0" stroke={sw.dark} stroke-width="0.6" />
    <rect x="52" y="4" width="6" height="4" fill="#f6e3b0" stroke={sw.dark} stroke-width="0.6" />
    {#each [24, 36, 58, 70] as x (x)}
      <rect {x} y="15" width="6" height="6" fill="#f6e3b0" stroke={sw.dark} stroke-width="0.6" />
    {/each}
    <path d="M8 30V18H16M84 18H92V30" fill="none" stroke={line} stroke-width="1.2" />
  {:else if color === 'purple'}
    <!-- Passenger coach: a clerestory roof over a row of windows. -->
    <rect x="9" y="11" width="82" height="19" rx="2" fill={sw.fill} stroke={sw.dark} stroke-width="1.2" />
    <path d="M8 12Q10 7 20 7H80Q90 7 92 12Z" fill={sw.dark} />
    <rect x="24" y="4" width="52" height="4" rx="1.5" fill={sw.dark} />
    {#each [15, 25, 35, 45, 55, 65, 75] as x (x)}
      <rect {x} y="14" width="7" height="8" rx="2" fill="#f6e3b0" stroke={sw.dark} stroke-width="0.6" />
    {/each}
    <line x1="10" y1="25" x2="90" y2="25" stroke="#f3d27a" stroke-width="1" />
  {:else if color === 'white'}
    <!-- Refrigerator car: boarded sides, ice hatches on the roof. -->
    <rect x="9" y="9" width="82" height="21" fill={sw.fill} stroke={sw.dark} stroke-width="1.2" />
    <rect x="7.5" y="7" width="85" height="2.6" rx="1" fill="#8f877a" />
    <rect x="13" y="4.5" width="9" height="2.6" fill="#8f877a" />
    <rect x="78" y="4.5" width="9" height="2.6" fill="#8f877a" />
    {#each [13, 16, 19, 22, 25, 28, 31, 69, 72, 75, 78, 81, 84, 87] as x (x)}
      <line x1={x} y1="10" x2={x} y2="29.5" stroke="#c9c1b2" stroke-width="0.6" />
    {/each}
    <rect x="38" y="11" width="24" height="18" fill="#ece5d6" stroke={sw.dark} stroke-width="1" />
    <rect x="57" y="18" width="3" height="4" fill={sw.dark} />
  {:else if color === 'orange'}
    <!-- Stock car: slatted sides. -->
    <rect x="9" y="9" width="82" height="21" fill="#3a2410" stroke={sw.dark} stroke-width="1.2" />
    {#each [11, 15.5, 20, 24.5] as y (y)}
      <rect x="9.6" {y} width="80.8" height="2.6" fill={sw.fill} />
    {/each}
    {#each [20, 31, 42, 58, 69, 80] as x (x)}
      <rect x={x - 1} y="9" width="2" height="21" fill={sw.fill} stroke={sw.dark} stroke-width="0.4" />
    {/each}
    <rect x="7.5" y="6.6" width="85" height="3" rx="1" fill={sw.dark} />
  {:else if color === 'yellow'}
    <!-- Gondola with a load of lumber. -->
    {#each [0, 1, 2] as row (row)}
      <rect x={14 + row * 2} y={13 - row * 3.4} width={72 - row * 4} height="3.4" fill="#b98a4a" stroke="#6b4a1f" stroke-width="0.6" />
    {/each}
    <rect x="9" y="16" width="82" height="14" fill={sw.fill} stroke={sw.dark} stroke-width="1.2" />
    {#each [19, 29, 39, 50, 61, 71, 81] as x (x)}
      <line x1={x} y1="16.5" x2={x} y2="29.5" stroke={sw.dark} stroke-width="0.7" />
    {/each}
    <line x1="10" y1="19.5" x2="90" y2="19.5" stroke="#fff" stroke-width="1" opacity="0.4" />
  {:else}
    <!-- Locomotive: boiler, stack, domes, cab and cowcatcher, on big drivers. -->
    <path d="M2 34L12 24H16V34Z" fill="#5a4325" stroke={line} stroke-width="0.8" />
    <rect x="14" y="14" width="52" height="15" rx="7.5" fill="#26201b" stroke={line} stroke-width="1" />
    <rect x="12" y="13.5" width="5" height="16" rx="2" fill="#b08a3e" />
    <path d="M21 14V5H19V2H29V5H27V14Z" fill="#26201b" stroke={line} stroke-width="0.8" />
    <rect x="18.5" y="1" width="11" height="2" fill="#b08a3e" />
    <rect x="36" y="9" width="8" height="6" rx="3" fill="#b08a3e" stroke={line} stroke-width="0.6" />
    <rect x="49" y="10" width="7" height="5" rx="2.5" fill="#26201b" stroke={line} stroke-width="0.6" />
    <circle cx="13" cy="10" r="2.8" fill="#f6e3b0" stroke={line} stroke-width="0.8" />
    <rect x="64" y="5" width="25" height="25" fill="#7a2a1f" stroke={line} stroke-width="1" />
    <rect x="62" y="3" width="29" height="3" rx="1" fill={line} />
    <rect x="69" y="9" width="9" height="8" rx="1" fill="#f6e3b0" stroke={line} stroke-width="0.6" />
    <line x1="14" y1="21.5" x2="64" y2="21.5" stroke="#b08a3e" stroke-width="1" />
    <rect x="12" y="29" width="80" height="3" fill={line} />
    {#each [32, 49] as x (x)}
      <circle cx={x} cy="33" r="7" fill={line} />
      <circle cx={x} cy="33" r="5" fill="none" stroke="#b08a3e" stroke-width="1" />
      <circle cx={x} cy="33" r="1.4" fill="#b08a3e" />
    {/each}
    <line x1="32" y1="35" x2="49" y2="35" stroke="#b08a3e" stroke-width="1.6" />
    <circle cx="20" cy="36.5" r="3.6" fill={line} />
    <circle cx="78" cy="36.5" r="3.6" fill={line} />
    <circle cx="86" cy="36.5" r="3.6" fill={line} />
  {/if}
  {#if color !== 'locomotive'}
    {@render frame()}
  {/if}
</g>
