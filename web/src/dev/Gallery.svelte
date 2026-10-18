<script lang="ts">
  import Board from '../board/Board.svelte';
  import TicketCard from '../cards/TicketCard.svelte';
  import TrainCard from '../cards/TrainCard.svelte';
  import { CARD_COLORS } from '../board/palette';
  import { layoutOnly, type BoardModel, type Layout } from '../board/layout';
  import usa from '../../../ticket_to_ride/data/layout/usa.json';
  import mini from '../../../ticket_to_ride/data/layout/mini.json';

  // From the map TOMLs; a wired board takes these from /api/boards/{map}.
  const ROUTE_POINTS: Record<string, number[]> = { usa: [1, 2, 4, 7, 10, 15], mini: [1, 2, 4, 7, 10] };
  const layouts: Record<string, Layout> = { usa: usa as unknown as Layout, mini: mini as unknown as Layout };
  let map = $state(new URLSearchParams(location.hash.split('?')[1]).get('map') ?? 'usa');
  const model: BoardModel = $derived(layoutOnly(layouts[map]!, ROUTE_POINTS[map]!));

  // A plausible mid-game: a few routes each, the other halves of their doubles closed.
  const claims: Record<string, [string, string, number][]> = {
    usa: [
      ['Seattle', 'Helena', 0], ['Helena', 'Denver', 0], ['Denver', 'Kansas City', 0], ['Kansas City', 'Saint Louis', 0],
      ['New York', 'Washington', 1], ['Washington', 'Raleigh', 1], ['Raleigh', 'Atlanta', 1], ['Atlanta', 'Miami', 1], ['Chicago', 'Pittsburgh', 1],
    ],
    mini: [
      ['Seattle', 'Helena', 0], ['Helena', 'Denver', 0], ['Salt Lake City', 'Denver', 1], ['Las Vegas', 'Los Angeles', 1], ['Los Angeles', 'Phoenix', 1],
    ],
  };
  const game = $derived.by(() => {
    const owner: (number | null)[] = model.segments.map(() => null);
    const closed = model.segments.map(() => false);
    for (const [a, b, seat] of claims[map] ?? []) {
      const seg = model.segments.find((s) => (s.a === a && s.b === b) || (s.a === b && s.b === a) );
      if (!seg) continue;
      owner[seg.index] = seat;
      if (seg.sibling !== null) closed[seg.sibling] = true;
    }
    return { owner, closed };
  });
  let highlight = $state<string[]>([]);
  const tickets: Record<string, [string, string, number, 'open' | 'done' | 'failed', number | null][]> = {
    usa: [
      ['Los Angeles', 'New York', 21, 'open', 9],
      ['Seattle', 'Los Angeles', 9, 'open', 4],
      ['Denver', 'Pittsburgh', 11, 'done', null],
      ['Duluth', 'Houston', 8, 'failed', null],
    ],
    mini: [
      ['Vancouver', 'El Paso', 14, 'open', 6],
      ['Helena', 'Los Angeles', 8, 'done', null],
      ['Winnipeg', 'Phoenix', 13, 'open', null],
    ],
  };
</script>

<main>
  <header>
    <h1>Board</h1>
    <nav>
      {#each Object.keys(layouts) as m (m)}
        <button class:on={map === m} onclick={() => (map = m)}>{m}</button>
      {/each}
      <button onmouseenter={() => (highlight = map === 'usa' ? ['Los Angeles', 'New York'] : ['Vancouver', 'El Paso'])} onmouseleave={() => (highlight = [])}>hover: ticket</button>
    </nav>
  </header>
  {#key map}
    <Board {model} owner={game.owner} closed={game.closed} scores={[37, 52]} {highlight} />
  {/key}
  <section class="row">
    {#each CARD_COLORS as c, i (c)}
      <TrainCard color={c} count={(i % 3) + 1} />
    {/each}
    <TrainCard color="locomotive" count={2} />
  </section>
  <section class="row">
    {#each tickets[map] ?? [] as [a, b, points, status, remaining] (a + b)}
      <TicketCard layout={model.layout} {a} {b} {points} {status} {remaining} onhover={(c) => (highlight = c ?? [])} />
    {/each}
  </section>
</main>

<style>
  .row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 16px;
  }
  main {
    padding: 16px 24px;
    max-width: 1600px;
    margin: 0 auto;
  }
  header {
    display: flex;
    align-items: baseline;
    gap: 24px;
    color: var(--paper);
    margin-bottom: 12px;
  }
  button {
    font: inherit;
    padding: 4px 12px;
    border-radius: 4px;
    border: 1px solid var(--gilt);
    background: transparent;
    color: var(--paper);
    cursor: pointer;
  }
  button.on {
    background: var(--gilt);
    color: var(--ink);
  }
</style>
