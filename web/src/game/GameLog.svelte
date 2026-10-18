<!--
  The game so far, in the server's plain English, newest at the bottom.
-->
<script lang="ts">
  import type { LogEntry } from '../api/protocol';
  import { seatColor } from '../board/palette';

  let { log, current = null }: { log: LogEntry[]; current?: number | null } = $props();

  let list = $state<HTMLOListElement>();
  $effect(() => {
    void log.length;
    if (list && current === null) list.scrollTop = list.scrollHeight;
  });
</script>

<ol class="log" bind:this={list}>
  {#each log as entry (entry.ply)}
    <li class:current={entry.ply === current}>
      <span class="seat-dot" style="background: {seatColor(entry.seat).fill}"></span>
      <span>{entry.text}</span>
    </li>
  {/each}
</ol>

<style>
  .log {
    list-style: none;
    margin: 0;
    padding: 4px 10px;
    overflow-y: auto;
    font-size: 13.5px;
    line-height: 1.35;
    height: 100%;
  }
  li {
    display: flex;
    gap: 7px;
    align-items: baseline;
    padding: 2px 0;
  }
  li.current {
    background: rgb(201 163 90 / 0.3);
  }
  .seat-dot {
    flex: none;
  }
</style>
