<!--
  The back of a train card or a destination ticket, for the draw piles.
-->
<script lang="ts">
  let { kind, width = 64 }: { kind: 'train' | 'ticket'; width?: number } = $props();
  const ticket = $derived(kind === 'ticket');
</script>

{#if ticket}
  <svg class="back" viewBox="0 0 170 112" {width} aria-hidden="true">
    <rect x="1" y="1" width="168" height="110" rx="6" fill="#efe0b8" stroke="#4a3220" stroke-width="1.4" />
    <rect x="8" y="8" width="154" height="96" rx="4" fill="none" stroke="#5c1d16" stroke-width="2" />
    <rect x="13" y="13" width="144" height="86" rx="3" fill="none" stroke="#c9a35a" stroke-width="1.2" />
    <text x="85" y="52" text-anchor="middle">Destination</text>
    <text x="85" y="76" text-anchor="middle" class="sub">Ticket</text>
  </svg>
{:else}
  <svg class="back" viewBox="0 0 100 140" {width} aria-hidden="true">
    <defs>
      <pattern id="card-back-lattice" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="10" height="10" fill="#5c1d16" />
        <path d="M0 0H10M0 0V10" stroke="#7a2a20" stroke-width="2" />
      </pattern>
    </defs>
    <rect x="1" y="1" width="98" height="138" rx="8" fill="url(#card-back-lattice)" stroke="#2c120c" stroke-width="1.4" />
    <rect x="7" y="7" width="86" height="126" rx="5" fill="none" stroke="#c9a35a" stroke-width="1.6" />
    <circle cx="50" cy="70" r="24" fill="#2c120c" stroke="#c9a35a" stroke-width="1.6" />
    <path d="M36 76H64V72H60V62H52V58H42V66H38V72H36Z" fill="#c9a35a" />
    <circle cx="42" cy="79" r="3.5" fill="#c9a35a" />
    <circle cx="57" cy="79" r="3.5" fill="#c9a35a" />
  </svg>
{/if}

<style>
  .back {
    display: block;
    height: auto;
    filter: drop-shadow(0 2px 3px rgb(0 0 0 / 0.35));
  }
  text {
    font-family: 'Alegreya SC', serif;
    font-weight: 700;
    font-size: 19px;
    fill: #5c1d16;
  }
  .sub {
    font-size: 15px;
    fill: #6a5039;
  }
</style>
