<script lang="ts">
  import type { Component } from 'svelte';

  let route = $state(location.hash);
  $effect(() => {
    const onhash = () => (route = location.hash);
    addEventListener('hashchange', onhash);
    return () => removeEventListener('hashchange', onhash);
  });

  // The board gallery is a development page: it reads the layout files directly, which the
  // production client never does (it reads /api/boards/{map}).
  const gallery: Promise<{ default: Component }> | null = import.meta.env.DEV
    ? import('./dev/Gallery.svelte')
    : null;
</script>

{#if gallery && (route === '' || route.startsWith('#/dev'))}
  {#await gallery then { default: Gallery }}
    <Gallery />
  {/await}
{/if}
