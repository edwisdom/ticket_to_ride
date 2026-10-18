<!--
  People and bots in one Bradley-Terry fit, on the H3 = 0 scale.
-->
<script lang="ts">
  import { api } from '../api/http';
  import type { Leaderboard, MapName } from '../api/protocol';
  import { elo, MAP_NAMES } from '../format';
  import { me } from '../player.svelte';
  import { href } from '../router';

  let { map }: { map: MapName } = $props();

  let board = $state<Leaderboard | null>(null);
  let failed = $state<string | null>(null);
  $effect(() => {
    board = null;
    api.leaderboard(map).then((b) => (board = b), (e: Error) => (failed = e.message));
  });
</script>

<div class="page">
  <section class="panel">
    <header>
      <h2>Leaderboard</h2>
      <div class="choices">
        {#each ['usa', 'mini'] as m (m)}
          <a class="button" class:on={map === m} href={href.leaderboard(m as MapName)}>{MAP_NAMES[m]}</a>
        {/each}
      </div>
    </header>
    <div class="body">
      {#if failed}
        <p class="error">{failed}</p>
      {:else if !board}
        <p class="muted">Tallying…</p>
      {:else if !board.standings.length}
        <p class="muted">{board.note ?? 'Nobody is rated on this map yet.'}</p>
      {:else}
        <table class="ledger">
          <thead>
            <tr><th class="num">#</th><th>Player</th><th class="num">Elo</th><th class="num">95% interval</th><th class="num">Comparisons</th><th class="num">Won</th></tr>
          </thead>
          <tbody>
            {#each board.standings as s, i (s.player.id)}
              <tr class:me={s.player.id === me.player?.id} class:bot={s.player.kind === 'bot'}>
                <td class="num">{i + 1}</td>
                <td>
                  {#if s.player.kind === 'human'}<a href={href.profile(s.player.id)}>{s.player.name}</a>{:else}{s.player.name}{/if}
                  {#if s.player.kind === 'bot'}<span class="tag">bot</span>{/if}
                  {#if s.player.spec === board.anchor}<span class="tag">anchor</span>{/if}
                </td>
                <td class="num elo">{elo({ ...s.rating, lo: null, hi: null })}</td>
                <td class="num muted">{s.rating.lo === null || s.rating.hi === null ? '–' : elo(s.rating).split(' ').slice(1).join(' ')}</td>
                <td class="num">{s.rating.games}</td>
                <td class="num">{s.rating.score === null ? '–' : `${Math.round(s.rating.score * 100)}%`}</td>
              </tr>
            {/each}
          </tbody>
        </table>
        <p class="muted foot">
          {board.games} {board.n_players}-player games in {board.blocks} seed blocks. Elo on the {board.anchor.toUpperCase()} = 0 scale: arena games
          and rated human games in one fit.
        </p>
      {/if}
    </div>
  </section>
</div>

<style>
  .page {
    padding: 22px;
    max-width: 860px;
    width: 100%;
    margin: 0 auto;
  }
  .choices .button {
    border-radius: 0;
    border: 0;
    padding: 3px 12px;
    background: var(--cream);
    color: var(--ink);
  }
  .choices .button.on {
    background: var(--gilt);
    color: var(--ink);
  }
  .elo {
    font-weight: 700;
  }
  .tag {
    font: 600 11px var(--caps);
    color: var(--ink-soft);
    border: 1px solid var(--paper-line);
    border-radius: 3px;
    padding: 0 4px;
    margin-left: 6px;
  }
  tr.bot td:nth-child(2) {
    color: var(--ink-soft);
  }
  .foot {
    font-size: 13px;
    margin: 10px 0 0;
  }
</style>
