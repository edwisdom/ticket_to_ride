<script lang="ts">
  import type { Component } from 'svelte';
  import { me } from './player.svelte';
  import { href, parse } from './router';
  import GameScreen from './screens/GameScreen.svelte';
  import Leaderboard from './screens/Leaderboard.svelte';
  import Lobby from './screens/Lobby.svelte';
  import Profile from './screens/Profile.svelte';
  import ReviewScreen from './screens/ReviewScreen.svelte';

  let route = $state(parse(location.hash));
  $effect(() => {
    const onhash = () => (route = parse(location.hash));
    addEventListener('hashchange', onhash);
    return () => removeEventListener('hashchange', onhash);
  });

  // The board gallery is a development page: it reads the layout files directly, which the
  // production client never does (it reads /api/boards/{map}).
  const gallery: Promise<{ default: Component }> | null = import.meta.env.DEV
    ? import('./dev/Gallery.svelte')
    : null;
</script>

<div class="app">
  <header class="bar">
    <a class="mark" href={href.lobby()}>TTR <span>Lab</span></a>
    <nav>
      <a href={href.lobby()} class:on={route.name === 'lobby'}>Play</a>
      <a href={href.leaderboard()} class:on={route.name === 'leaderboard'}>Leaderboard</a>
      {#if me.player}
        <a href={href.profile()} class:on={route.name === 'profile'}>{me.player.name}</a>
      {/if}
    </nav>
  </header>
  <main>
    {#if route.name === 'lobby'}
      <Lobby />
    {:else if route.name === 'game'}
      {#key route.id}
        <GameScreen id={route.id} />
      {/key}
    {:else if route.name === 'review'}
      {#key route.id}
        <ReviewScreen id={route.id} start={route.at} />
      {/key}
    {:else if route.name === 'profile'}
      {#key route.id}
        <Profile id={route.id ?? me.player?.id ?? null} />
      {/key}
    {:else if route.name === 'leaderboard'}
      <Leaderboard map={route.map} />
    {:else if route.name === 'dev' && gallery}
      {#await gallery then { default: Gallery }}
        <Gallery />
      {/await}
    {:else}
      <p class="lost">Nothing here. <a href={href.lobby()}>Back to the lobby</a>.</p>
    {/if}
  </main>
</div>

<style>
  .app {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
  }
  .bar {
    display: flex;
    align-items: center;
    gap: 28px;
    height: 44px;
    padding: 0 20px;
    background: linear-gradient(#3a1410, #2a0f0b);
    border-bottom: 2px solid var(--gilt);
    box-shadow: 0 2px 10px rgb(0 0 0 / 0.4);
    flex: none;
  }
  .mark {
    font-family: var(--display);
    font-size: 21px;
    color: var(--gilt-soft);
    text-decoration: none;
    letter-spacing: 0.04em;
  }
  .mark span {
    color: var(--cream);
  }
  nav {
    display: flex;
    gap: 18px;
  }
  nav a {
    font: 600 14px var(--caps);
    letter-spacing: 0.04em;
    color: var(--paper-deep);
    text-decoration: none;
    padding: 4px 2px;
    border-bottom: 2px solid transparent;
  }
  nav a.on,
  nav a:hover {
    color: var(--cream);
    border-bottom-color: var(--gilt);
  }
  main {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .lost {
    color: var(--paper);
    padding: 40px;
  }
</style>
