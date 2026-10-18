<!--
  The lobby: who you are, a new game against a bot or between two, and your games.
-->
<script lang="ts">
  import { api, HttpError } from '../api/http';
  import type { Agent, GameSummary, MapName } from '../api/protocol';
  import { elo, MAP_NAMES, outcome, ratingOn, versus, when } from '../format';
  import { me, remember } from '../player.svelte';
  import { href } from '../router';

  const MAPS: MapName[] = ['usa', 'mini'];

  let agents = $state<Agent[]>([]);
  let games = $state<GameSummary[]>([]);
  let error = $state<string | null>(null);
  let busy = $state(false);

  let name = $state('');
  let map = $state<MapName>('usa');
  let opponent = $state('h3');
  let rated = $state(true);
  let seat = $state<number | null>(null);
  let watchMap = $state<MapName>('mini');
  let watch = $state<[string, string]>(['h3', 'h4']);

  const fail = (e: unknown) => (error = e instanceof HttpError || e instanceof Error ? e.message : String(e));

  $effect(() => {
    api.agents().then((a) => (agents = a), fail);
  });
  $effect(() => {
    const id = me.player?.id;
    if (id) api.games(id).then((g) => (games = g), fail);
  });

  const unfinished = $derived(games.filter((g) => g.status === 'active'));
  const finished = $derived(games.filter((g) => g.status === 'finished').slice(0, 12));

  async function join(e: SubmitEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    busy = true;
    try {
      remember(await api.newPlayer(name.trim()));
    } catch (err) {
      fail(err);
    } finally {
      busy = false;
    }
  }

  async function start(body: Parameters<typeof api.newGame>[0]) {
    busy = true;
    error = null;
    try {
      const game = await api.newGame(body);
      location.hash = href.game(game.id);
    } catch (err) {
      fail(err);
      busy = false;
    }
  }

  function play() {
    if (!me.player) return;
    start({ mode: 'play', map, opponent, player_id: me.player.id, rated, ...(rated ? {} : { seat }) });
  }
</script>

<div class="lobby">
  {#if !me.player}
    <section class="panel welcome">
      <header><h2>Welcome aboard</h2></header>
      <form class="body" onsubmit={join}>
        <p>What should the conductor call you? Your games and rating go under this name.</p>
        <input bind:value={name} placeholder="Your name" maxlength="40" required />
        <button disabled={busy || !name.trim()}>Board the train</button>
      </form>
    </section>
  {/if}

  <section class="panel">
    <header><h2>Play a bot</h2></header>
    <div class="body form">
      <div class="row">
        <span class="label">Map</span>
        <div class="choices">
          {#each MAPS as m (m)}
            <button class:on={map === m} onclick={() => (map = m)}>{MAP_NAMES[m]}</button>
          {/each}
        </div>
      </div>
      <div class="agents">
        {#each agents as a (a.spec)}
          <button class="agent" class:on={opponent === a.spec} onclick={() => (opponent = a.spec)}>
            <strong>{a.name}</strong>
            <span class="rating">{a.anchor ? '0 (anchor)' : elo(ratingOn(a.ratings, map))}</span>
            <span class="desc">{a.description}</span>
          </button>
        {/each}
      </div>
      <div class="row">
        <label><input type="checkbox" bind:checked={rated} /> Rated</label>
        {#if rated}
          <span class="muted">Your seat is drawn at random: moving first is worth a little.</span>
        {:else}
          <div class="choices">
            {#each [[null, 'Random'], [0, 'First'], [1, 'Second']] as [s, label] (label)}
              <button class:on={seat === s} onclick={() => (seat = s as number | null)}>{label}</button>
            {/each}
          </div>
        {/if}
      </div>
      <div class="row">
        <button disabled={busy || !me.player} onclick={play}>Start the game</button>
      </div>
    </div>
  </section>

  <section class="panel">
    <header><h2>Watch two bots</h2></header>
    <div class="body form">
      <div class="row">
        <span class="label">Map</span>
        <div class="choices">
          {#each MAPS as m (m)}
            <button class:on={watchMap === m} onclick={() => (watchMap = m)}>{MAP_NAMES[m]}</button>
          {/each}
        </div>
      </div>
      <div class="row">
        <select bind:value={watch[0]}>
          {#each agents as a (a.spec)}<option value={a.spec}>{a.name}</option>{/each}
        </select>
        <span>vs</span>
        <select bind:value={watch[1]}>
          {#each agents as a (a.spec)}<option value={a.spec}>{a.name}</option>{/each}
        </select>
      </div>
      <p class="muted">Every hand and ticket is visible, with each bot's reasoning beside the board.</p>
      <div class="row">
        <button disabled={busy} onclick={() => start({ mode: 'spectate', map: watchMap, agents: watch })}>Watch</button>
      </div>
    </div>
  </section>

  {#if me.player}
    <section class="panel games">
      <header><h2>Your games</h2></header>
      <div class="body">
        {#if !games.length}
          <p class="muted">None yet.</p>
        {/if}
        {#if unfinished.length}
          <h3>Unfinished</h3>
          <table class="ledger">
            <tbody>
              {#each unfinished as g (g.id)}
                <tr>
                  <td>{MAP_NAMES[g.map]}</td>
                  <td>{versus(g)}</td>
                  <td>{g.rated ? 'rated' : 'unrated'}</td>
                  <td class="num">move {g.ply}</td>
                  <td class="muted">{when(g.created)}</td>
                  <td class="num"><a class="button quiet" href={href.game(g.id)}>Resume</a></td>
                </tr>
              {/each}
            </tbody>
          </table>
        {/if}
        {#if finished.length}
          <h3>Finished</h3>
          <table class="ledger">
            <tbody>
              {#each finished as g (g.id)}
                <tr>
                  <td>{MAP_NAMES[g.map]}</td>
                  <td>{versus(g)}</td>
                  <td>{outcome(g) ?? ''}</td>
                  <td class="muted">{g.finished ? when(g.finished) : ''}</td>
                  <td class="num"><a class="button quiet" href={href.review(g.id)}>Review</a></td>
                </tr>
              {/each}
            </tbody>
          </table>
        {/if}
      </div>
    </section>
  {/if}

  {#if error}
    <p class="error">{error}</p>
  {/if}
</div>

<style>
  .lobby {
    display: grid;
    grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr);
    gap: 18px;
    padding: 22px;
    max-width: 1180px;
    width: 100%;
    margin: 0 auto;
    align-items: start;
  }
  .welcome,
  .games,
  .error {
    grid-column: 1 / -1;
  }
  .welcome form {
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
  }
  .welcome p {
    flex-basis: 100%;
    margin: 0 0 4px;
  }
  .form {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  .label {
    font: 700 13px var(--caps);
    color: var(--ink-soft);
    width: 36px;
  }
  .agents {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 8px;
  }
  .agent {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: baseline;
    gap: 2px 8px;
    text-align: left;
    background: var(--cream);
    color: var(--ink);
    border: 1px solid var(--paper-line);
    padding: 8px 10px;
  }
  .agent strong {
    font: 700 16px var(--caps);
  }
  .agent .rating {
    font-variant-numeric: tabular-nums;
    font-size: 13px;
    color: var(--ink-soft);
  }
  .agent .desc {
    grid-column: 1 / -1;
    font-weight: 400;
    font-size: 13px;
    color: var(--ink-soft);
  }
  .agent.on {
    border-color: var(--oxblood);
    box-shadow: 0 0 0 2px var(--gilt) inset;
    background: #fff8e6;
  }
  .games h3 {
    font-size: 14px;
    color: var(--ink-soft);
    margin: 10px 0 4px;
  }
  .games h3:first-child {
    margin-top: 0;
  }
  p {
    margin: 0;
  }
  .error {
    color: #ffb4a8;
  }
</style>
