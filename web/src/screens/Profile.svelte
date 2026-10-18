<!--
  A player: their rating on each map with its interval, their record against each opponent,
  and their finished games.
-->
<script lang="ts">
  import { api } from '../api/http';
  import type { Profile } from '../api/protocol';
  import { elo, MAP_NAMES, plural, when } from '../format';
  import { forget, me } from '../player.svelte';
  import { href } from '../router';

  let { id }: { id: string | null } = $props();

  let profile = $state<Profile | null>(null);
  let failed = $state<string | null>(null);
  $effect(() => {
    if (id) api.profile(id).then((p) => (profile = p), (e: Error) => (failed = e.message));
  });

  const self = $derived(id !== null && id === me.player?.id);
  const pct = (n: number, of: number) => (of ? `${Math.round((100 * n) / of)}%` : '–');
</script>

<div class="page">
  {#if !id}
    <p class="notice">Pick a name in the <a href={href.lobby()}>lobby</a> first.</p>
  {:else if failed}
    <p class="notice error">{failed}</p>
  {:else if !profile}
    <p class="notice">Looking you up…</p>
  {:else}
    <section class="panel">
      <header>
        <h2>{profile.player.name}</h2>
        {#if self}<button class="quiet signout" onclick={() => (forget(), (location.hash = href.lobby()))}>Not you?</button>{/if}
      </header>
      <div class="body ratings">
        {#each profile.ratings as r (r.map)}
          <div class="rating">
            <span class="map">{MAP_NAMES[r.map]}</span>
            <span class="elo">{elo(r.rating)}</span>
            <span class="muted">{plural(r.rating.games, 'comparison')}{r.rating.score !== null ? ` · ${pct(r.rating.score * 100, 100)} won` : ''}</span>
          </div>
        {/each}
        {#if !profile.ratings.length}
          <p class="muted">No rating yet: finish a rated game. A rating needs a path to H3, so play H3 or have the arena run.</p>
        {/if}
        <p class="muted scale">Elo anchored on H3, with a 95% interval; the leaderboard has everyone on it.</p>
      </div>
    </section>

    <section class="panel">
      <header><h2>Against each opponent</h2></header>
      <div class="body">
        <table class="ledger">
          <thead>
            <tr>
              <th>Map</th><th>Opponent</th><th class="num">Games</th><th class="num">Won</th><th class="num">Lost</th>
              <th class="num">Drawn</th><th class="num">Win rate</th><th class="num">Mean score</th><th class="num">Theirs</th>
            </tr>
          </thead>
          <tbody>
            {#each profile.record as r (r.map + r.opponent.id)}
              <tr>
                <td>{MAP_NAMES[r.map]}</td>
                <td>{r.opponent.name}</td>
                <td class="num">{r.games}</td>
                <td class="num">{r.wins}</td>
                <td class="num">{r.losses}</td>
                <td class="num">{r.draws}</td>
                <td class="num">{pct(r.wins, r.games)}</td>
                <td class="num">{r.mean_score.toFixed(1)}</td>
                <td class="num">{r.mean_opponent_score.toFixed(1)}</td>
              </tr>
            {:else}
              <tr><td colspan="9" class="muted">No finished games yet.</td></tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <header><h2>Finished games</h2></header>
      <div class="body">
        <table class="ledger">
          <tbody>
            {#each profile.games as g (g.game_id)}
              <tr>
                <td>{MAP_NAMES[g.map]}</td>
                <td>vs {g.opponent.name}</td>
                <td class:won={g.result === 'win'} class:lost={g.result === 'loss'}>{g.result}</td>
                <td class="num">{g.score}–{g.opponent_score}</td>
                <td>{g.rated ? 'rated' : 'unrated'}</td>
                <td>seat {g.seat + 1}</td>
                <td class="muted">{when(g.finished)}</td>
                <td class="num"><a class="button quiet" href={href.review(g.game_id)}>Review</a></td>
              </tr>
            {:else}
              <tr><td class="muted">None yet.</td></tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>
  {/if}
</div>

<style>
  .page {
    display: flex;
    flex-direction: column;
    gap: 18px;
    padding: 22px;
    max-width: 980px;
    width: 100%;
    margin: 0 auto;
  }
  .notice {
    color: var(--paper);
  }
  .notice a {
    color: var(--gilt-soft);
  }
  .signout {
    color: var(--cream);
    border-color: var(--gilt);
    padding: 2px 10px;
  }
  .ratings {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 36px;
    align-items: baseline;
  }
  .rating {
    display: flex;
    flex-direction: column;
  }
  .map {
    font: 700 13px var(--caps);
    color: var(--ink-soft);
  }
  .elo {
    font: 700 26px var(--sans);
    font-variant-numeric: tabular-nums;
  }
  .scale {
    flex-basis: 100%;
    margin: 0;
    font-size: 13px;
  }
  .won {
    color: var(--good);
    font-weight: 700;
  }
  .lost {
    color: var(--bad);
  }
</style>
